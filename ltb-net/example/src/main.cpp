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

#include "example_client.hpp"
#include "example_server.hpp"

auto main(int argc, const char* argv[]) -> int {
    std::string host_address = "0.0.0.0:50051";
    auto        client_only  = false;
    auto        server_only  = false;

    auto display_help_message = [] {
        std::cout << "run_example [host_address] [options...]"
                  << "host_address  - Defaults to '0.0.0.0:50051' if not provided\n"
                  << "--client,-c   - Run only the client\n"
                  << "--server,-s   - Run only the server\n"
                  << "--help,-h     - Display this message\n"
                  << std::endl;
    };

    for (auto i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            display_help_message();
        } else if (arg == "--client" || arg == "-c") {
            client_only = true;
        } else if (arg == "--server" || arg == "-s") {
            server_only = true;
        } else {
            host_address = arg;
        }
    }

    // using Service = ltb::example::ChatRoom;

    if (client_only && server_only) {
        std::cerr << "ERROR: the [--client,-c] flag and [--server,-s] flag cannot be used together" << std::endl;
        display_help_message();
        return EXIT_FAILURE;
    }

    if (client_only) {
        ltb::example::ExampleClient client(host_address);
        client.run();
        return EXIT_SUCCESS;
    }

    if (server_only) {
        ltb::example::ExampleServer server(host_address);
        std::cout << "Press enter to exit" << std::endl;
        std::cin.ignore();
        server.shutdown();
        return EXIT_SUCCESS;
    }

    ltb::example::ExampleServer server;
    ltb::example::ExampleClient client(server.grpc_server());
    client.run();
    server.shutdown();

    return EXIT_SUCCESS;
}
