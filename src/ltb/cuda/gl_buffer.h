// ///////////////////////////////////////////////////////////////////////////////////////
//                                                                           |________|
//  Copyright (c) 2019 CloudNC Ltd - All Rights Reserved                        |  |
//                                                                              |__|
//        ____                                                                .  ||
//       / __ \                                                               .`~||$$$$
//      | /  \ \         /$$$$$$  /$$                           /$$ /$$   /$$  /$$$$$$$
//      \ \ \ \ \       /$$__  $$| $$                          | $$| $$$ | $$ /$$__  $$
//    / / /  \ \ \     | $$  \__/| $$  /$$$$$$  /$$   /$$  /$$$$$$$| $$$$| $$| $$  \__/
//   / / /    \ \__    | $$      | $$ /$$__  $$| $$  | $$ /$$__  $$| $$ $$ $$| $$
//  / / /      \__ \   | $$      | $$| $$  \ $$| $$  | $$| $$  | $$| $$  $$$$| $$
// | | / ________ \ \  | $$    $$| $$| $$  | $$| $$  | $$| $$  | $$| $$\  $$$| $$    $$
//  \ \_/ ________/ /  |  $$$$$$/| $$|  $$$$$$/|  $$$$$$/|  $$$$$$$| $$ \  $$|  $$$$$$/
//   \___/ ________/    \______/ |__/ \______/  \______/  \_______/|__/  \__/ \______/
//
// ///////////////////////////////////////////////////////////////////////////////////////
#pragma once

// project
#include "ltb/cuda/cuda_check.hpp"

// external
#include <Corrade/Containers/ArrayViewStl.h>
#include <Magnum/GL/Buffer.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

// standard
#include <vector>

namespace ltb {
namespace cuda {

/// @brief Maps an allocated OpenGL buffer for use with CUDA
template <typename T>
class GLBuffer {
public:
    explicit GLBuffer(Magnum::GL::Buffer gl_buffer = Magnum::GL::Buffer{});
    ~GLBuffer();

    auto set_stream(cudaStream_t stream) -> void;

    auto map_for_cuda() -> void;
    auto unmap_from_cuda() -> void;

    auto cuda_buffer(std::size_t* size) const -> T*;
    auto gl_buffer() const -> Magnum::GL::Buffer const&;

    auto size() const -> std::size_t;

private:
    Magnum::GL::Buffer     gl_buffer_;
    cudaGraphicsResource_t graphics_resource_ = nullptr;
    cudaStream_t           stream_            = nullptr;
    bool                   mapped_for_cuda_   = false;
};

template <typename T>
GLBuffer<T>::GLBuffer(Magnum::GL::Buffer gl_buffer) : gl_buffer_(std::move(gl_buffer)) {
    LTB_CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&graphics_resource_, gl_buffer_.id(), cudaGraphicsRegisterFlagsNone));
}

template <typename T>
GLBuffer<T>::~GLBuffer() {
    LTB_CUDA_CHECK(cudaGraphicsUnregisterResource(graphics_resource_));
}

template <typename T>
auto GLBuffer<T>::set_stream(cudaStream_t stream) -> void {
    stream_ = stream;
}

template <typename T>
auto GLBuffer<T>::map_for_cuda() -> void {
    if (!mapped_for_cuda_) {
        LTB_CUDA_CHECK(cudaGraphicsMapResources(1, &graphics_resource_, stream_));
        mapped_for_cuda_ = true;
    }
}

template <typename T>
auto GLBuffer<T>::unmap_from_cuda() -> void {
    if (mapped_for_cuda_) {
        LTB_CUDA_CHECK(cudaGraphicsUnmapResources(1, &graphics_resource_, stream_));
        mapped_for_cuda_ = false;
    }
}

template <typename T>
auto GLBuffer<T>::cuda_buffer(std::size_t* size) const -> T* {
    T*          dev_ptr   = nullptr;
    std::size_t byte_size = 0;
    LTB_CUDA_CHECK(
        cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&dev_ptr), &byte_size, graphics_resource_));
    *size = byte_size / sizeof(T);
    return dev_ptr;
}

template <typename T>
auto GLBuffer<T>::gl_buffer() const -> Magnum::GL::Buffer const& {
    return gl_buffer_;
}

template <typename T>
auto GLBuffer<T>::size() const -> std::size_t {
    return gl_buffer().size();
}

} // namespace cuda
} // namespace ltb
