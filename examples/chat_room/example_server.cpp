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

// project
#include "example_service.hpp"
#include "ltb/net/server/async_server_data.hpp"

// external
#include <grpc++/server_builder.h>

namespace ltb::example {
namespace {

auto set_callbacks(ltb::net::AsyncServer<ChatRoom::AsyncService>* async_server, ExampleService* service) -> void {
    async_server->unary_rpc(&ChatRoom::AsyncService::RequestDispatchAction,
                            [service](Action const& action, ltb::net::AsyncUnaryWriter<util::Result> writer) {
                                util::Result response;
                                auto         status = service->handle_action(action, &response);
                                writer.finish(response, status);
                            });
}

} // namespace

ExampleServer::ExampleServer(std::string const& host_address) : async_server_(host_address) {
    run_thread_ = std::thread([this] {
        ExampleService service;
        set_callbacks(&async_server_, &service);
        async_server_.run();
    });
}

ExampleServer::~ExampleServer() {
    shutdown();
    run_thread_.join();
}

auto ExampleServer::grpc_server() -> grpc::Server& {
    return async_server_.grpc_server();
}

auto ExampleServer::shutdown() -> void {
    async_server_.shutdown();
}

} // namespace ltb::example
