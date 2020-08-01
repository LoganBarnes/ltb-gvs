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
#include <vector>

namespace ltb::cuda {

namespace {

auto to_cu_deviceptr(void const* ptr) -> CUdeviceptr {
    return reinterpret_cast<CUdeviceptr>(ptr);
}

struct ScopedDeviceMemory {
    explicit ScopedDeviceMemory(std::size_t size_in_bytes)
        : data_(thrust::raw_pointer_cast(thrust::device_malloc(size_in_bytes)), cudaFree) {}

    // NOLINTNEXTLINE(google-explicit-constructor, hicpp-explicit-conversions)
    operator CUdeviceptr() const { return reinterpret_cast<CUdeviceptr>(data_.get()); }

    // NOLINTNEXTLINE(google-explicit-constructor, hicpp-explicit-conversions)
    operator void*() const { return data_.get(); }

private:
    std::shared_ptr<void> data_;
};

} // namespace

auto OptiX::init() -> std::shared_ptr<CUstream_st> {
    CUstream_st* raw_stream = nullptr;
    LTB_CUDA_CHECK(cudaStreamCreate(&raw_stream));
    auto stream = std::shared_ptr<CUstream_st>(raw_stream, cudaStreamDestroy);

    {
        // Initialize CUDA
        LTB_CUDA_CHECK(cudaFree(nullptr));

        // Initialize OptiX
        LTB_OPTIX_CHECK(optixInit());
    }

    return stream;
}

auto OptiX::make_context(const OptixDeviceContextOptions& options) -> std::shared_ptr<OptixDeviceContext_t> {
    OptixDeviceContext_t* context = nullptr;
    LTB_OPTIX_CHECK(optixDeviceContextCreate(nullptr, &options, &context));
    return std::shared_ptr<OptixDeviceContext_t>(context, optixDeviceContextDestroy);
}

auto OptiX::build_geometry(std::shared_ptr<OptixDeviceContext_t> const& context,
                           OptixAccelBuildOptions const&                accel_build_options)
    -> std::shared_ptr<OptixTraversableHandle> {

    // Triangle build input: simple list of three vertices
    const std::vector<float3> vertices = {
        {-0.5f, -0.5f, 0.0f},
        {0.5f, -0.5f, 0.0f},
        {0.0f, 0.5f, 0.0f},
    };
    auto verts_byte_size = vertices.size() * sizeof(float3);
    auto device_vertices = ScopedDeviceMemory(verts_byte_size);
    LTB_CUDA_CHECK(cudaMemcpy(device_vertices, vertices.data(), verts_byte_size, cudaMemcpyHostToDevice));

    CUdeviceptr d_vertices = device_vertices;

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
    LTB_OPTIX_CHECK(optixAccelComputeMemoryUsage(context.get(),
                                                 &accel_build_options,
                                                 &triangle_input,
                                                 1, // Number of build inputs
                                                 &gas_buffer_sizes));

    auto temp_buffer_gas   = ScopedDeviceMemory(gas_buffer_sizes.tempSizeInBytes);
    auto gas_output_buffer = ScopedDeviceMemory(gas_buffer_sizes.outputSizeInBytes);

    auto gas_handle = std::shared_ptr<OptixTraversableHandle>(new OptixTraversableHandle(0),
                                                              [context, gas_output_buffer](auto* ptr) { delete ptr; });

    LTB_OPTIX_CHECK(optixAccelBuild(context.get(),
                                    nullptr, // CUDA stream
                                    &accel_build_options,
                                    &triangle_input,
                                    1, // num build inputs
                                    temp_buffer_gas,
                                    gas_buffer_sizes.tempSizeInBytes,
                                    gas_output_buffer,
                                    gas_buffer_sizes.outputSizeInBytes,
                                    gas_handle.get(),
                                    nullptr, // emitted property list
                                    0 // num emitted properties
                                    ));
    return gas_handle;
}

} // namespace ltb::cuda
