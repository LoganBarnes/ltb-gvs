// ///////////////////////////////////////////////////////////////////////////////////////
//                                                                           |________|
//  Copyright (c) 2020 CloudNC Ltd - All Rights Reserved                        |  |
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

// generated
#include <protos/chat/chat_room.grpc.pb.h>

// external
#include <grpc++/server.h>

namespace ltb::example {

class ExampleService : public ChatRoom::Service {
public:
    ~ExampleService() override;

    grpc::Status DispatchAction(grpc::ServerContext* context, Action const* request, util::Result* response) override;

    grpc::Status
    PokeUser(grpc::ServerContext* context, grpc::ServerReader<User::Id>* reader, util::Result* response) override;

    grpc::Status SearchMessages(grpc::ServerContext*                 context,
                                google::protobuf::StringValue const* request,
                                grpc::ServerWriter<UserMessage>*     writer) override;

    grpc::Status GetMessages(grpc::ServerContext*                                         context,
                             grpc::ServerReaderWriter<ChatMessageResult, ChatMessage_Id>* stream) override;

    grpc::Status GetUpdates(grpc::ServerContext*           context,
                            google::protobuf::Empty const* request,
                            grpc::ServerWriter<Update>*    writer) override;
};

class ExampleServer {
public:
    explicit ExampleServer(std::string const& host_address = "");

    auto grpc_server() -> grpc::Server&;
    auto shutdown() -> void;

private:
    ExampleService                service_;
    std::unique_ptr<grpc::Server> server_;
};

} // namespace ltb::example
