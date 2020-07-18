// ///////////////////////////////////////////////////////////////////////////////////////
// LTB Utilities
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

// external
#include <optix_stubs.h>

// standard
#include <exception>
#include <sstream>
#include <string>

#define LTB_OPTIX_CHECK(val) ::ltb::optix::check((val), #val, __FILE__, __LINE__)

namespace ltb {
namespace optix {

template <typename T>
void check(T result, char const* const func, const char* const file, int const line) {
    if (result != OPTIX_SUCCESS) {
        std::stringstream error_str;
        error_str << "OPTIX error at " << file << ":" << line;
        error_str << " code=" << static_cast<unsigned int>(result) << "(" << optixGetErrorString(result) << ") ";
        error_str << "\"" << func << "\"";

        throw std::runtime_error(error_str.str());
    }
}

} // namespace optix
} // namespace ltb
