// ///////////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2020 Logan Barnes - All Rights Reserved
// ///////////////////////////////////////////////////////////////////////////////////////
#include "optix.hpp"

// project
#include "ltb/cuda/cuda_check.hpp"
#include "ltb/cuda/optix_check.hpp"

// external
#include <optix_function_table_definition.h> // can only exist once in a translation unit
#include <optix_stubs.h>

namespace ltb::cuda {

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

} // namespace ltb
