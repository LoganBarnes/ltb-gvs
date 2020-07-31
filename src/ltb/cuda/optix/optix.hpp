// ///////////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2020 Logan Barnes - All Rights Reserved
// ///////////////////////////////////////////////////////////////////////////////////////
#pragma once

// external
#include <optix_types.h>
#include <driver_types.h>

// standard
#include <memory>

namespace ltb::cuda {

class OptiX {
public:
    static auto init() -> std::shared_ptr<CUstream_st>;
    static auto make_context(OptixDeviceContextOptions const& options) -> std::shared_ptr<OptixDeviceContext_t>;
};

} // namespace ltb
