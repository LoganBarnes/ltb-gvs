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
#include "example_renderable.hpp"

// project
#include "ltb/gvs/display/magnum_conversions.hpp"

// external
#include <Magnum/GL/Mesh.h>
#include <Magnum/GL/Shader.h>
#include <Magnum/SceneGraph/Camera.h>
#include <Magnum/SceneGraph/MatrixTransformation3D.h>
#include <Magnum/SceneGraph/Object.h>

using namespace Magnum;

namespace ltb::example {

class ExampleRenderable::ExampleDrawable : public SceneGraph::Drawable3D {
public:
    explicit ExampleDrawable(SceneGraph::Object<SceneGraph::MatrixTransformation3D>& object,
                             SceneGraph::DrawableGroup3D*                            group,
                             unsigned                                                intersect_id,
                             GL::Mesh&                                               mesh,
                             GridShader&                                             shader)
        : SceneGraph::Drawable3D{object, group},
          object_(object),
          mesh_(mesh),
          intersect_id_(intersect_id),
          shader_(shader) {}

    ~ExampleDrawable() override = default;

    auto update_display_info(const gvs::DisplayInfo& display_info) -> void {
        object_.setTransformation(gvs::to_magnum(display_info.transformation));
        coloring_      = display_info.coloring;
        uniform_color_ = gvs::to_magnum<Magnum::Color3>(display_info.uniform_color);
        shading_       = display_info.shading;
    }

    auto update_time(float seconds) -> void { sim_time_secs_ = seconds; }

private:
    void draw(Matrix4 const& transformation_matrix, SceneGraph::Camera3D& camera) override {
        shader_.set_projection_from_world_matrix(camera.projectionMatrix() * transformation_matrix)
            .set_coloring(coloring_)
            .set_uniform_color(uniform_color_)
            .set_shading(shading_)
            .set_id(intersect_id_)
            .set_time(sim_time_secs_)
            .draw(mesh_);
    }

    SceneGraph::Object<SceneGraph::MatrixTransformation3D>& object_;
    GL::Mesh&                                               mesh_;

    Color3 uniform_color_ = {gvs::default_uniform_color[0], //
                             gvs::default_uniform_color[1],
                             gvs::default_uniform_color[2]};

    gvs::Coloring coloring_     = gvs::default_coloring;
    gvs::Shading  shading_      = gvs::default_shading;
    unsigned      intersect_id_ = 0u;

    float sim_time_secs_ = 0.f;

    GridShader& shader_;
};

ExampleRenderable::ExampleRenderable() {
    mesh_.setPrimitive(MeshPrimitive::Points);
    mesh_.setCount(10000);
}

ExampleRenderable::~ExampleRenderable() = default;

auto ExampleRenderable::init(SceneGraph::Object<SceneGraph::MatrixTransformation3D>& object,
                             SceneGraph::DrawableGroup3D*                            group,
                             unsigned                                                intersect_id) -> void {
    drawable_ = std::make_shared<ExampleDrawable>(object, group, intersect_id, mesh_, shader_);
}

auto ExampleRenderable::drawable() const -> SceneGraph::Drawable3D* {
    return drawable_.get();
}
auto ExampleRenderable::set_display_info(gvs::DisplayInfo const & /*display_info*/) -> void {}

auto ExampleRenderable::resize(Vector2i const & /*viewport*/) -> void {}

auto ExampleRenderable::update(util::Duration const& sim_time) -> void {
    drawable_->update_time(util::to_seconds<float>(sim_time));
}

} // namespace ltb::example
