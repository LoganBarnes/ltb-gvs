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

#include "gl_buffer.h"
#include "gl_buffer_image.h"

namespace ltb {
namespace cuda {

template <typename T>
class GLBufferMapGuard {
public:
    explicit GLBufferMapGuard(GLBuffer<T>& interop_buffer) : interop_buffer(interop_buffer) {
        interop_buffer.map_for_cuda();
    }
    ~GLBufferMapGuard() { interop_buffer.unmap_from_cuda(); }

private:
    GLBuffer<T>& interop_buffer;
};

// Allows the guard to be created without having to specify the template type
template <typename T>
GLBufferMapGuard<T> make_gl_buffer_map_guard(GLBuffer<T>& interop_buffer) {
    return GLBufferMapGuard<T>(interop_buffer);
}

template <typename T>
class GLBufferImageMapGuard {
public:
    explicit GLBufferImageMapGuard(GLBufferImage<T>& interop_buffer) : interop_buffer(interop_buffer) {
        interop_buffer.map_for_cuda();
    }
    ~GLBufferImageMapGuard() { interop_buffer.unmap_from_cuda(); }

private:
    GLBufferImage<T>& interop_buffer;
};

// Allows the guard to be created without having to specify the template type
template <typename T>
GLBufferImageMapGuard<T> make_gl_buffer_map_guard(GLBufferImage<T>& interop_buffer) {
    return GLBufferImageMapGuard<T>(interop_buffer);
}

} // namespace cuda
} // namespace ltb
