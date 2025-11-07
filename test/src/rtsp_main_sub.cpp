#include "stream_viewer.hpp"

int main() {
    
    std::string user = "admin";
    std::string pass = "Panto2025";
    std::string ip = "192.168.0.101";        
    int port = 554;
    std::string stream_type = "main";         // Cambia entre "main" o "sub"

    // Crear y ejecutar el visor
    StreamViewer viewer(user, pass, ip, port, stream_type);
    viewer.run();

    return 0;
}