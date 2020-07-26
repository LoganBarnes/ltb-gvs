// ///////////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2020 Logan Barnes - All Rights Reserved
// ///////////////////////////////////////////////////////////////////////////////////////
#include "optix.hpp"

// project
#include "common/params.hpp"
#include "ltb/cuda/buffer_map_guard.h"
#include "ltb/cuda/cuda_check.hpp"
#include "ltb/cuda/optix_check.hpp"
#include "ltb/gvs_cuda_includes.hpp"

// external
#include <cuda/helpers.h>
#include <nvrtc.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>
#include <thrust/device_vector.h>

// standard
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// NVRTC compiler options
#define CUDA_NVRTC_OPTIONS                                                                                             \
    "-std=c++17", "-arch", "compute_61", "-use_fast_math", "-lineinfo", "-default-device", "-rdc", "true", "-D__x86_64",

#define STRINGIFY(x) STRINGIFY2(x)
#define STRINGIFY2(x) #x
#define LINE_STR STRINGIFY(__LINE__)

#define NVRTC_CHECK_ERROR(func)                                                                                        \
    do {                                                                                                               \
        nvrtcResult code = func;                                                                                       \
        if (code != NVRTC_SUCCESS) {                                                                                   \
            throw std::runtime_error("ERROR: " __FILE__ "(" LINE_STR "): " + std::string(nvrtcGetErrorString(code)));  \
        }                                                                                                              \
    } while (false)

namespace app {
namespace {

template <typename T>
struct SbtRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<RayGenData>   RayGenSbtRecord;
typedef SbtRecord<MissData>     MissSbtRecord;
typedef SbtRecord<HitGroupData> HitGroupSbtRecord;

std::string g_nvrtc_log;

void context_log_cb(unsigned int level, const char* tag, const char* message, void* /*cbdata */) {
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: " << message << std::endl;
}

auto get_ptx_from_cu_string(const std::string&       cu_source,
                            const std::string&       name,
                            std::vector<std::string> include_dirs,
                            const char**             log_string) -> std::string {
    // Create program
    nvrtcProgram prog = 0;
    NVRTC_CHECK_ERROR(nvrtcCreateProgram(&prog, cu_source.c_str(), name.c_str(), 0, nullptr, nullptr));

    include_dirs.emplace_back(ltb::paths::cuda_include_dir());
    include_dirs.emplace_back(ltb::paths::optix_include_dir());
    include_dirs.emplace_back(ltb::paths::gvs_root() + "external");

    // Gather NVRTC options
    auto include_dir_options = std::vector<std::string>{};

    // Collect include dirs
    for (const std::string& dir : include_dirs) {
        include_dir_options.push_back("-I" + dir);
    }

    // Collect NVRTC options
    auto compiler_options = std::vector<std::string>{CUDA_NVRTC_OPTIONS};

    auto options = std::vector<const char*>{};

    for (const auto& option : include_dir_options) {
        options.emplace_back(option.c_str());
    }
    for (const auto& option : compiler_options) {
        options.emplace_back(option.c_str());
    }

    // JIT compile CU to PTX
    const nvrtcResult compile_res = nvrtcCompileProgram(prog, static_cast<int>(options.size()), options.data());

    // Retrieve log output
    size_t log_size = 0;
    NVRTC_CHECK_ERROR(nvrtcGetProgramLogSize(prog, &log_size));
    g_nvrtc_log.resize(log_size);
    if (log_size > 1) {
        NVRTC_CHECK_ERROR(nvrtcGetProgramLog(prog, &g_nvrtc_log[0]));
        if (log_string) {
            *log_string = g_nvrtc_log.c_str();
        }
    }
    if (compile_res != NVRTC_SUCCESS) {
        throw std::runtime_error("NVRTC Compilation failed.\n" + g_nvrtc_log);
    }

    // Retrieve PTX code
    std::string ptx;
    size_t      ptx_size = 0;
    NVRTC_CHECK_ERROR(nvrtcGetPTXSize(prog, &ptx_size));
    ptx.resize(ptx_size);
    NVRTC_CHECK_ERROR(nvrtcGetPTX(prog, &ptx[0]));

    // Cleanup
    NVRTC_CHECK_ERROR(nvrtcDestroyProgram(&prog));
    return ptx;
}

} // namespace

struct OptiX::Data {
    cudaStream_t stream = nullptr;

    OptixDeviceContext context = nullptr;

    OptixAccelBuildOptions accel_options = {};

    OptixModule module = nullptr;

    OptixTraversableHandle gas_handle;
    CUdeviceptr            d_gas_output_buffer;

    OptixShaderBindingTable sbt = {};

    OptixProgramGroup raygen_prog_group   = nullptr;
    OptixProgramGroup miss_prog_group     = nullptr;
    OptixProgramGroup hitgroup_prog_group = nullptr;
    OptixPipeline     pipeline            = nullptr;
};

OptiX::OptiX() : data_(std::make_unique<Data>()) {}

OptiX::~OptiX() = default;

auto OptiX::init(const std::string& programs_str) -> void {
    LTB_CUDA_CHECK(cudaStreamCreate(&data_->stream));

    char log[2048]; // For error reporting from OptiX creation functions

    {
        // Initialize CUDA
        LTB_CUDA_CHECK(cudaFree(nullptr));

        // Initialize OptiX
        LTB_OPTIX_CHECK(optixInit());

        CUcontext                 cu_ctx  = nullptr; // null means take the current context
        OptixDeviceContextOptions options = {};
        options.logCallbackFunction       = &context_log_cb;
        options.logCallbackLevel          = 4;

        LTB_OPTIX_CHECK(optixDeviceContextCreate(cu_ctx, &options, &data_->context));
    }

    //
    // accel handling
    //
    {
        // Use default options for simplicity.  In a real use case we would want to
        // enable compaction, etc
        data_->accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
        data_->accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;

        // Triangle build input: simple list of three vertices
        const std::vector<float3> vertices = {
            {-0.5f, -0.5f, 0.0f},
            {0.5f, -0.5f, 0.0f},
            {0.0f, 0.5f, 0.0f},
        };
        const thrust::device_vector<float3> device_vertices = {vertices.begin(), vertices.end()};
        const auto d_vertices = reinterpret_cast<CUdeviceptr>(thrust::raw_pointer_cast(device_vertices.data()));

        // Our build input is a simple list of non-indexed triangle vertices
        const uint32_t  triangle_input_flags[1]    = {OPTIX_GEOMETRY_FLAG_NONE};
        OptixBuildInput triangle_input             = {};
        triangle_input.type                        = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        triangle_input.triangleArray.vertexFormat  = OPTIX_VERTEX_FORMAT_FLOAT3;
        triangle_input.triangleArray.numVertices   = static_cast<uint32_t>(vertices.size());
        triangle_input.triangleArray.vertexBuffers = &d_vertices;
        triangle_input.triangleArray.flags         = triangle_input_flags;
        triangle_input.triangleArray.numSbtRecords = 1;

        OptixAccelBufferSizes gas_buffer_sizes;
        LTB_OPTIX_CHECK(optixAccelComputeMemoryUsage(data_->context,
                                                     &data_->accel_options,
                                                     &triangle_input,
                                                     1, // Number of build inputs
                                                     &gas_buffer_sizes));
        CUdeviceptr d_temp_buffer_gas;
        LTB_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer_gas), gas_buffer_sizes.tempSizeInBytes));
        LTB_CUDA_CHECK(
            cudaMalloc(reinterpret_cast<void**>(&data_->d_gas_output_buffer), gas_buffer_sizes.outputSizeInBytes));

        LTB_OPTIX_CHECK(optixAccelBuild(data_->context,
                                        0, // CUDA stream
                                        &data_->accel_options,
                                        &triangle_input,
                                        1, // num build inputs
                                        d_temp_buffer_gas,
                                        gas_buffer_sizes.tempSizeInBytes,
                                        data_->d_gas_output_buffer,
                                        gas_buffer_sizes.outputSizeInBytes,
                                        &data_->gas_handle,
                                        nullptr, // emitted property list
                                        0 // num emitted properties
                                        ));

        // We can now free the scratch space buffer used during build and the vertex
        // inputs, since they are not needed by our trivial shading method
        LTB_CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer_gas)));
    }

    //
    // Create module
    //
    OptixPipelineCompileOptions pipeline_compile_options = {};
    {
        OptixModuleCompileOptions module_compile_options = {};
        module_compile_options.maxRegisterCount          = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
        module_compile_options.optLevel                  = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        module_compile_options.debugLevel                = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

        pipeline_compile_options.usesMotionBlur        = false;
        pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        pipeline_compile_options.numPayloadValues      = 3;
        pipeline_compile_options.numAttributeValues    = 3;
#ifdef DEBUG // Enables debug exceptions during optix launches. This may incur significant performance cost and should only be done during development.
        pipeline_compile_options.exceptionFlags
            = OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
#else
        pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#endif
        pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
        pipeline_compile_options.usesPrimitiveTypeFlags = static_cast<unsigned>(OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE);

        //        auto ptx = getPtxFromCuString(programs_str);
        auto   ptx        = get_ptx_from_cu_string(programs_str, "", {ltb::paths::project_root() + "src"}, nullptr);
        size_t sizeof_log = sizeof(log);

        LTB_OPTIX_CHECK(optixModuleCreateFromPTX(data_->context,
                                                 &module_compile_options,
                                                 &pipeline_compile_options,
                                                 ptx.c_str(),
                                                 ptx.size(),
                                                 log,
                                                 &sizeof_log,
                                                 &data_->module));
    }

    //
    // Create program groups
    //
    {
        OptixProgramGroupOptions program_group_options = {}; // Initialize to zeros

        OptixProgramGroupDesc raygen_prog_group_desc    = {}; //
        raygen_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygen_prog_group_desc.raygen.module            = data_->module;
        raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
        size_t sizeof_log                               = sizeof(log);
        LTB_OPTIX_CHECK(optixProgramGroupCreate(data_->context,
                                                &raygen_prog_group_desc,
                                                1, // num program groups
                                                &program_group_options,
                                                log,
                                                &sizeof_log,
                                                &data_->raygen_prog_group));

        OptixProgramGroupDesc miss_prog_group_desc  = {};
        miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prog_group_desc.miss.module            = data_->module;
        miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
        sizeof_log                                  = sizeof(log);
        LTB_OPTIX_CHECK(optixProgramGroupCreate(data_->context,
                                                &miss_prog_group_desc,
                                                1, // num program groups
                                                &program_group_options,
                                                log,
                                                &sizeof_log,
                                                &data_->miss_prog_group));

        OptixProgramGroupDesc hitgroup_prog_group_desc        = {};
        hitgroup_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hitgroup_prog_group_desc.hitgroup.moduleCH            = data_->module;
        hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
        sizeof_log                                            = sizeof(log);
        LTB_OPTIX_CHECK(optixProgramGroupCreate(data_->context,
                                                &hitgroup_prog_group_desc,
                                                1, // num program groups
                                                &program_group_options,
                                                log,
                                                &sizeof_log,
                                                &data_->hitgroup_prog_group));
    }

    //
    // Link pipeline
    //
    {
        const uint32_t    max_trace_depth = 1;
        OptixProgramGroup program_groups[]
            = {data_->raygen_prog_group, data_->miss_prog_group, data_->hitgroup_prog_group};

        OptixPipelineLinkOptions pipeline_link_options = {};
        pipeline_link_options.maxTraceDepth            = max_trace_depth;
        pipeline_link_options.debugLevel               = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
        size_t sizeof_log                              = sizeof(log);
        LTB_OPTIX_CHECK(optixPipelineCreate(data_->context,
                                            &pipeline_compile_options,
                                            &pipeline_link_options,
                                            program_groups,
                                            sizeof(program_groups) / sizeof(program_groups[0]),
                                            log,
                                            &sizeof_log,
                                            &data_->pipeline));

        OptixStackSizes stack_sizes = {};
        for (auto& prog_group : program_groups) {
            LTB_OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes));
        }

        uint32_t direct_callable_stack_size_from_traversal;
        uint32_t direct_callable_stack_size_from_state;
        uint32_t continuation_stack_size;
        LTB_OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes,
                                                   max_trace_depth,
                                                   0, // maxCCDepth
                                                   0, // maxDCDEpth
                                                   &direct_callable_stack_size_from_traversal,
                                                   &direct_callable_stack_size_from_state,
                                                   &continuation_stack_size));
        LTB_OPTIX_CHECK(optixPipelineSetStackSize(data_->pipeline,
                                                  direct_callable_stack_size_from_traversal,
                                                  direct_callable_stack_size_from_state,
                                                  continuation_stack_size,
                                                  1 // maxTraversableDepth
                                                  ));
    }

    //
    // Set up shader binding table
    //
    {
        CUdeviceptr  raygen_record;
        const size_t raygen_record_size = sizeof(RayGenSbtRecord);
        LTB_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&raygen_record), raygen_record_size));
        RayGenSbtRecord rg_sbt;
        LTB_OPTIX_CHECK(optixSbtRecordPackHeader(data_->raygen_prog_group, &rg_sbt));
        LTB_CUDA_CHECK(
            cudaMemcpy(reinterpret_cast<void*>(raygen_record), &rg_sbt, raygen_record_size, cudaMemcpyHostToDevice));

        CUdeviceptr miss_record;
        size_t      miss_record_size = sizeof(MissSbtRecord);
        LTB_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&miss_record), miss_record_size));
        MissSbtRecord ms_sbt;
        ms_sbt.data = {0.3f, 0.1f, 0.2f};
        LTB_OPTIX_CHECK(optixSbtRecordPackHeader(data_->miss_prog_group, &ms_sbt));
        LTB_CUDA_CHECK(
            cudaMemcpy(reinterpret_cast<void*>(miss_record), &ms_sbt, miss_record_size, cudaMemcpyHostToDevice));

        CUdeviceptr hitgroup_record;
        size_t      hitgroup_record_size = sizeof(HitGroupSbtRecord);
        LTB_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&hitgroup_record), hitgroup_record_size));
        HitGroupSbtRecord hg_sbt;
        LTB_OPTIX_CHECK(optixSbtRecordPackHeader(data_->hitgroup_prog_group, &hg_sbt));
        LTB_CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(hitgroup_record),
                                  &hg_sbt,
                                  hitgroup_record_size,
                                  cudaMemcpyHostToDevice));

        data_->sbt.raygenRecord                = raygen_record;
        data_->sbt.missRecordBase              = miss_record;
        data_->sbt.missRecordStrideInBytes     = sizeof(MissSbtRecord);
        data_->sbt.missRecordCount             = 1;
        data_->sbt.hitgroupRecordBase          = hitgroup_record;
        data_->sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
        data_->sbt.hitgroupRecordCount         = 1;
    }
}

auto OptiX::destroy() -> void {
    LTB_CUDA_CHECK(cudaFree(reinterpret_cast<void*>(data_->sbt.raygenRecord)));
    LTB_CUDA_CHECK(cudaFree(reinterpret_cast<void*>(data_->sbt.missRecordBase)));
    LTB_CUDA_CHECK(cudaFree(reinterpret_cast<void*>(data_->sbt.hitgroupRecordBase)));
    LTB_CUDA_CHECK(cudaFree(reinterpret_cast<void*>(data_->d_gas_output_buffer)));

    LTB_OPTIX_CHECK(optixPipelineDestroy(data_->pipeline));
    LTB_OPTIX_CHECK(optixProgramGroupDestroy(data_->hitgroup_prog_group));
    LTB_OPTIX_CHECK(optixProgramGroupDestroy(data_->miss_prog_group));
    LTB_OPTIX_CHECK(optixProgramGroupDestroy(data_->raygen_prog_group));
    LTB_OPTIX_CHECK(optixModuleDestroy(data_->module));

    LTB_OPTIX_CHECK(optixDeviceContextDestroy(data_->context));

    LTB_CUDA_CHECK(cudaStreamDestroy(data_->stream));
}

auto OptiX::stream() const -> cudaStream_t {
    return data_->stream;
}

auto OptiX::launch(ltb::cuda::GLBufferImage<uchar4>& output_buffer) -> void {
    auto guard = ltb::cuda::make_gl_buffer_map_guard(output_buffer);

    Params params;
    params.image        = output_buffer.cuda_buffer();
    params.image_width  = output_buffer.gl_buffer_image().size().x();
    params.image_height = output_buffer.gl_buffer_image().size().y();
    params.handle       = data_->gas_handle;
    params.cam_eye      = {0.0f, 0.0f, 2.0f};

    auto look_at      = float3{0.0f, 0.0f, 0.0f};
    auto up           = float3{0.0f, 1.0f, 3.0f};
    auto fovy         = 45.0f;
    auto aspect_ratio = Magnum::Vector2{output_buffer.gl_buffer_image().size()}.aspectRatio();

    params.cam_w = look_at - params.cam_eye; // Do not normalize W -- it implies focal length
    float wlen   = length(params.cam_w);
    params.cam_u = normalize(cross(params.cam_w, up));
    params.cam_v = normalize(cross(params.cam_u, params.cam_w));

    float vlen = wlen * tanf(0.5f * fovy * M_PIf / 180.0f);
    params.cam_v *= vlen;
    float ulen = vlen * aspect_ratio;
    params.cam_u *= ulen;

    CUdeviceptr d_param;
    LTB_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_param), sizeof(Params)));
    LTB_CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_param), &params, sizeof(params), cudaMemcpyHostToDevice));

    LTB_OPTIX_CHECK(optixLaunch(data_->pipeline,
                                data_->stream,
                                d_param,
                                sizeof(Params),
                                &data_->sbt,
                                params.image_width,
                                params.image_height,
                                /*depth=*/1));
    LTB_CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace app
