// ///////////////////////////////////////////////////////////////////////////////////////
// LTB Geometry Visualization Server
// Copyright (c) 2020 Logan Barnes - All Rights Reserved
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
// ///////////////////////////////////////////////////////////////////////////////////////
#pragma once

// project
#include "ltb/gvs/core/custom_renderable.hpp"
#include "ltb/gvs/core/types.hpp"

// external
#include <Magnum/SceneGraph/Drawable.h>

namespace ltb::gvs {

class OpenglRenderable : public CustomRenderable {
public:
    ~OpenglRenderable() override = 0;

    // CustomRenderable start
    auto configure_gui(DisplayInfo* display_info) -> bool override = 0;
    // CustomRenderable end

    virtual auto init_gl_types(Magnum::SceneGraph::Object<Magnum::SceneGraph::MatrixTransformation3D>& object,
                               Magnum::SceneGraph::DrawableGroup3D*                                    group,
                               unsigned intersect_id) -> void
        = 0;

    [[nodiscard]] virtual auto drawable() const -> Magnum::SceneGraph::Drawable3D* = 0;

    virtual auto set_display_info(DisplayInfo const& display_info) -> void = 0;

    virtual auto resize(Magnum::Vector2i const& viewport) -> void = 0;
};

inline OpenglRenderable::~OpenglRenderable() = default;

} // namespace ltb::gvs
