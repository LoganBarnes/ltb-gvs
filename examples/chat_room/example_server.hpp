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

// generated
#include <protos/chat/chat_room.grpc.pb.h>

// external
#include <grpc++/server.h>

// standard
#include <thread>

namespace ltb::example {

class ExampleServer {
public:
    explicit ExampleServer(std::string const& host_address = "");

    auto grpc_server() -> grpc::Server&;

    auto run() -> void;

    auto shutdown() -> void;

private:
    std::mutex                                   mutex_;
    ChatRoom::AsyncService                       service_;
    std::unique_ptr<grpc::ServerCompletionQueue> completion_queue_;
    std::unique_ptr<grpc::Server>                server_;
    std::thread                                  run_thread_;

    auto set_callbacks() -> void;
};

} // namespace ltb::example
