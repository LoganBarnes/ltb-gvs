// ///////////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2020 Logan Barnes - All Rights Reserved
// ///////////////////////////////////////////////////////////////////////////////////////
#pragma once

// project
#include "ltb/cuda/gl_buffer_image.hpp"

// standard
#include <memory>

namespace app {

class OptiX {
public:
    explicit OptiX();
    ~OptiX();

    auto init(const std::string& programs_str) -> void;
    auto destroy() -> void;

    auto stream() const -> cudaStream_t;

    auto launch(ltb::cuda::GLBufferImage<uchar4>& output_buffer) -> void;

private:
    struct Data;
    std::unique_ptr<Data> data_;
};

} // namespace app
