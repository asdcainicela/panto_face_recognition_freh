#include "stream_viewer.hpp"

int main() {
    
    std::string user = "admin";
    std::string pass = "Panto2025";
    std::string ip = "192.168.0.101";        
    int port = 554;
    std::string stream_type_main = "main";   
    std::string stream_type_sub = "sub";  

    // crear y ejecutar el visor
    StreamViewer viewer_main(user, pass, ip, port, stream_type_main);
    StreamViewer viewer_sub(user, pass, ip, port, stream_type_sub);
    viewer_main.run();
    viewer_sub.run();


    return 0;
}