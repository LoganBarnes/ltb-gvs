// ///////////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2020 Logan Barnes - All Rights Reserved
// ///////////////////////////////////////////////////////////////////////////////////////
#pragma once

// project
#include "ltb/cuda/gl_buffer_image.hpp"
#include "ltb/util/result.hpp"

// external
#include <driver_types.h>
#include <optix_types.h>

namespace ltb::cuda {

class OptiX {
public:
    explicit OptiX();
    ~OptiX();

    auto init() -> util::Result<OptiX*>;
    auto create_module(OptixPipelineCompileOptions const& pipeline_compile_options,
                       OptixModuleCompileOptions const&   module_compile_options,
                       std::string const&                 programs_str,
                       std::vector<std::string> const&    include_dirs) -> ltb::util::Result<OptiX*>;

    auto init(const std::string& programs_str) -> util::Result<OptiX*>;
    auto destroy() -> void;

    auto stream() const -> cudaStream_t;

    auto launch(GLBufferImage<uchar4>& output_buffer) -> void;

private:
    cudaStream_t stream_ = nullptr;

    OptixDeviceContext context_ = nullptr;

    OptixAccelBuildOptions accel_options_ = {};

    OptixModule module_ = nullptr;

    OptixTraversableHandle gas_handle_          = {};
    CUdeviceptr            d_gas_output_buffer_ = {};

    OptixShaderBindingTable sbt_ = {};

    OptixProgramGroup raygen_prog_group_   = nullptr;
    OptixProgramGroup miss_prog_group_     = nullptr;
    OptixProgramGroup hitgroup_prog_group_ = nullptr;
    OptixPipeline     pipeline_            = nullptr;
};

} // namespace ltb::cuda
