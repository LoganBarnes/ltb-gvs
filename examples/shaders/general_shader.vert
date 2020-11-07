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

// version will be inserted automagically

in int gl_VertexID;

uniform mat4  projection_from_world = mat4(1.f);
uniform int   grid_width            = 100;
uniform float grid_scale            = 0.1f;
uniform float time_secs             = 0.f;

layout(location = 0) out vec3 world_position;
layout(location = 1) out vec3 world_normal;
layout(location = 2) out vec2 texture_coordinates;


out gl_PerVertex
{
    vec4 gl_Position;
};

ivec2 grid_index(in int index_1d)
{
    ivec2 index_2d;
    index_2d.y = index_1d / grid_width;
    index_2d.x = index_1d - (grid_width * index_2d.y);
    return index_2d;
}

void main()
{
    ivec2 index_2d = grid_index(gl_VertexID);


    texture_coordinates = vec2(index_2d);

    world_position.xy   = texture_coordinates - grid_width * 0.5f;
    world_position.xy   *= grid_scale;

    world_position.z    = -5.f;
    world_position.z    += cos(time_secs + world_position.x);
    world_position.z    += sin(time_secs + world_position.y);
    world_position.z    *= 0.2f;

    world_normal        = vec3(0.f, 0.f, 1.f);

    gl_Position = projection_from_world * vec4(world_position, 1.f);
}
