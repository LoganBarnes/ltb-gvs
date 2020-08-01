// ///////////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2020 Logan Barnes - All Rights Reserved
// ///////////////////////////////////////////////////////////////////////////////////////
#pragma once

// project
#include "ltb/util/result.hpp"

// external
#include <driver_types.h>
#include <optix_types.h>

// standard
#include <memory>

namespace ltb::cuda {

class OptiX {
public:
    /// \brief
    /// \return
    static auto init() -> util::Result<std::shared_ptr<CUstream_st>>;

    /// \brief
    /// \param options
    /// \return
    static auto make_context(OptixDeviceContextOptions const& options)
        -> util::Result<std::shared_ptr<OptixDeviceContext_t>>;

    /// \brief
    /// \param context
    /// \param accel_build_options
    /// \return
    static auto build_geometry(std::shared_ptr<OptixDeviceContext_t> const& context,
                               OptixAccelBuildOptions const&                accel_build_options)
        -> util::Result<std::shared_ptr<OptixTraversableHandle>>;
};

} // namespace ltb::cuda
