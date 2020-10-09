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
#include "example_server.hpp"

#include <grpc++/server_builder.h>

namespace ltb::example {

ExampleServer::ExampleServer(std::string const& host_address) {
    grpc::ServerBuilder builder;
    if (!host_address.empty()) {
        // std::cout << "S: " << host_address << std::endl;
        builder.AddListeningPort(host_address, grpc::InsecureServerCredentials());
    }
    builder.RegisterService(&service_);
    completion_queue_ = builder.AddCompletionQueue();
    server_           = builder.BuildAndStart();
}

auto ExampleServer::grpc_server() -> grpc::Server& {
    return *server_;
}

auto ExampleServer::run() -> void {

    void* raw_tag                = {};
    bool  completed_successfully = {};

    while (completion_queue_->Next(&raw_tag, &completed_successfully)) {
        if (completed_successfully) {

        } else {

        }
    }
}

auto ExampleServer::shutdown() -> void {
    server_->Shutdown();
}

auto ExampleServer::set_callbacks() -> void {

    grpc::ServerContext                                         context;
    Action                                                      request;
    grpc::ServerAsyncResponseWriter<ltb::example::util::Result> response(&context);

    service_.RequestDispatchAction(&context,
                                   &request,
                                   &response,
                                   completion_queue_.get(),
                                   completion_queue_.get(),
                                   nullptr);
}

} // namespace ltb::example
