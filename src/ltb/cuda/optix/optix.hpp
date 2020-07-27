// ///////////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2020 Logan Barnes - All Rights Reserved
// ///////////////////////////////////////////////////////////////////////////////////////
#pragma once

// project
#include "ltb/cuda/gl_buffer_image.hpp"
#include "ltb/util/result.hpp"

// external
#include <optix_types.h>

// standard
#include <memory>

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
    struct Data;
    std::unique_ptr<Data> data_;
};

} // namespace ltb::cuda
