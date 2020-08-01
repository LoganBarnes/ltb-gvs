// ///////////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2020 Logan Barnes - All Rights Reserved
// ///////////////////////////////////////////////////////////////////////////////////////
#include "optix.hpp"

// project
#include "ltb/cuda/cuda_check.hpp"
#include "ltb/cuda/optix_check.hpp"

// external
#include <cuda_runtime_api.h>
#include <optix_function_table_definition.h> // can only exist once in a translation unit
#include <optix_stubs.h>
#include <thrust/device_malloc.h>

// standard
#include <thread>
#include <vector>

//#define DEBUG_LOGGING

namespace ltb::cuda {

namespace {

auto to_cu_deviceptr(void const* ptr) -> CUdeviceptr {
    return reinterpret_cast<CUdeviceptr>(ptr);
}

struct ScopedDeviceMemory {
    explicit ScopedDeviceMemory(std::shared_ptr<void> data) : data_(std::move(data)) {}

    // NOLINTNEXTLINE(google-explicit-constructor, hicpp-explicit-conversions)
    operator CUdeviceptr() const { return to_cu_deviceptr(data_.get()); }

    // NOLINTNEXTLINE(google-explicit-constructor, hicpp-explicit-conversions)
    operator void*() const { return data_.get(); }

private:
    std::shared_ptr<void> data_;
};

auto make_scoped_device_memory(std::size_t size_in_bytes) -> util::Result<ScopedDeviceMemory> {
    void* ptr = nullptr;
    LTB_SAFE_CUDA_CHECK(cudaMalloc(&ptr, size_in_bytes));
    return ScopedDeviceMemory(std::shared_ptr<void>(ptr, [](auto* _ptr) { LTB_CUDA_CHECK(cudaFree(_ptr)); }));
}

#define LTB_CHECK_RESULT(val)                                                                                          \
    if (!(val)) {                                                                                                      \
        return tl::make_unexpected((val).error());                                                                     \
    }

} // namespace

auto OptiX::init() -> util::Result<std::shared_ptr<CUstream_st>> {
    CUstream_st* raw_stream = nullptr;
    LTB_SAFE_CUDA_CHECK(cudaStreamCreate(&raw_stream))

#ifdef DEBUG_LOGGING
    std::cerr << "stream " << raw_stream << " created" << std::endl;
    auto stream = std::shared_ptr<CUstream_st>(raw_stream, [](auto* ptr) {
        std::cerr << "stream " << ptr << " destroyed" << std::endl;
        LTB_CUDA_CHECK(cudaStreamDestroy(ptr));
    });
#else
    auto stream = std::shared_ptr<CUstream_st>(raw_stream, cudaStreamDestroy);
#endif

    {
        // Initialize CUDA
        LTB_SAFE_CUDA_CHECK(cudaFree(nullptr));

        // Initialize OptiX
        LTB_SAFE_OPTIX_CHECK(optixInit());
    }

    return stream;
}

auto OptiX::make_context(const OptixDeviceContextOptions& options)
    -> util::Result<std::shared_ptr<OptixDeviceContext_t>> {

    OptixDeviceContext_t* context = nullptr;
    LTB_SAFE_OPTIX_CHECK(optixDeviceContextCreate(nullptr, &options, &context))

#ifdef DEBUG_LOGGING
    std::cerr << "context " << context << " created" << std::endl;
    return std::shared_ptr<OptixDeviceContext_t>(context, [](auto* ptr) {
        std::cerr << "context " << ptr << " destroyed" << std::endl;
        LTB_OPTIX_CHECK(optixDeviceContextDestroy(ptr));
    });
#else
    return std::shared_ptr<OptixDeviceContext_t>(context, optixDeviceContextDestroy);
#endif
}

auto OptiX::build_geometry(std::shared_ptr<OptixDeviceContext_t> const& context,
                           OptixAccelBuildOptions const&                accel_build_options)
    -> util::Result<std::shared_ptr<OptixTraversableHandle>> {

#ifdef DEBUG_LOGGING
    std::cout << "[" << std::this_thread::get_id() << "]: " << __FUNCTION__ << std::endl;
#endif

    // Triangle build input: simple list of three vertices
    const std::vector<float3> vertices = {
        {-0.5f, -0.5f, 0.0f},
        {0.5f, -0.5f, 0.0f},
        {0.0f, 0.5f, 0.0f},
    };
    auto verts_byte_size = vertices.size() * sizeof(float3);
    auto device_vertices = make_scoped_device_memory(verts_byte_size);
    LTB_CHECK_RESULT(device_vertices)
    LTB_SAFE_CUDA_CHECK(cudaMemcpy(device_vertices.value(), vertices.data(), verts_byte_size, cudaMemcpyHostToDevice));

    CUdeviceptr d_vertices = device_vertices.value();

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
    LTB_SAFE_OPTIX_CHECK(optixAccelComputeMemoryUsage(context.get(),
                                                      &accel_build_options,
                                                      &triangle_input,
                                                      1, // Number of build inputs
                                                      &gas_buffer_sizes))

    auto temp_buffer_gas = make_scoped_device_memory(gas_buffer_sizes.tempSizeInBytes);
    LTB_CHECK_RESULT(temp_buffer_gas)
    auto gas_output_buffer = make_scoped_device_memory(gas_buffer_sizes.outputSizeInBytes);
    LTB_CHECK_RESULT(gas_output_buffer)

#ifdef DEBUG_LOGGING
    auto gas_handle = std::shared_ptr<OptixTraversableHandle>(new OptixTraversableHandle(0),
                                                              [context, gas_output_buffer](auto* ptr) {
                                                                  std::cerr << "gas_handle " << ptr << " destroyed"
                                                                            << std::endl;
                                                                  delete ptr;
                                                              });
    std::cerr << "gas_handle " << gas_handle.get() << " created" << std::endl;
#else
    auto gas_handle = std::shared_ptr<OptixTraversableHandle>(new OptixTraversableHandle(0),
                                                              [context, gas_output_buffer](auto* ptr) { delete ptr; });
#endif

    LTB_SAFE_OPTIX_CHECK(optixAccelBuild(context.get(),
                                         nullptr, // CUDA stream
                                         &accel_build_options,
                                         &triangle_input,
                                         1, // num build inputs
                                         temp_buffer_gas.value(),
                                         gas_buffer_sizes.tempSizeInBytes,
                                         gas_output_buffer.value(),
                                         gas_buffer_sizes.outputSizeInBytes,
                                         gas_handle.get(),
                                         nullptr, // emitted property list
                                         0 // num emitted properties
                                         ))

    return gas_handle;
}

} // namespace ltb::cuda
