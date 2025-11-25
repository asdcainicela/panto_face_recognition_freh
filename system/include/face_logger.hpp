// ============= include/face_logger.hpp =============
/*
 * Face Data Logger - Sistema simplificado de registro de rostros
 * 
 * FORMATO JSON:
 * {
 *   "id": "a7f3k9m2",
 *   "timestamp": "2025-11-24 14:30:52.123",
 *   "age": 35,
 *   "gender": "Male",
 *   "company": "Freh",
 *   "emotion": "Happy",
 *   "embedding": [0.123, -0.456, 0.789, ...]  // 512 valores
 * }
 * 
 * AUTOR: PANTO System
 * FECHA: 2025
 */

#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <mutex>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <random>

struct FaceLogEntry {
    std::string id;                      // ID único (ej: "a7f3k9m2")
    std::string timestamp;               // Timestamp
    int age;                             // Edad
    std::string gender;                  // "Male" o "Female"
    std::string company;                 // "Freh"
    std::string emotion;                 // Emoción
    std::vector<float> embedding;        // 512 valores del rostro
    
    FaceLogEntry() 
        : id(""), age(0), gender("Unknown"), 
          company("Freh"), emotion("Unknown") {}
    
    // Generar ID aleatorio (alfanumérico)
    static std::string generate_id(int length = 8) {
        static const char chars[] = 
            "0123456789"
            "abcdefghijklmnopqrstuvwxyz";
        
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_int_distribution<> dis(0, sizeof(chars) - 2);
        
        std::string id;
        for (int i = 0; i < length; i++) {
            id += chars[dis(gen)];
        }
        return id;
    }
    
    // Generar timestamp actual
    static std::string current_timestamp() {
        auto now = std::chrono::system_clock::now();
        auto time_t_now = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;
        
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time_t_now), "%Y-%m-%d %H:%M:%S");
        ss << '.' << std::setfill('0') << std::setw(3) << ms.count();
        return ss.str();
    }
    
    // Convertir a JSON string
    std::string to_json() const {
        std::stringstream json;
        json << "  {\n";
        json << "    \"id\": \"" << id << "\",\n";
        json << "    \"timestamp\": \"" << timestamp << "\",\n";
        json << "    \"age\": " << age << ",\n";
        json << "    \"gender\": \"" << gender << "\",\n";
        json << "    \"company\": \"" << company << "\",\n";
        json << "    \"emotion\": \"" << emotion << "\",\n";
        json << "    \"embedding\": [";
        
        // Escribir embedding (512 valores)
        for (size_t i = 0; i < embedding.size(); i++) {
            json << std::fixed << std::setprecision(6) << embedding[i];
            if (i < embedding.size() - 1) json << ", ";
        }
        
        json << "]\n";
        json << "  }";
        return json.str();
    }
};

class FaceLogger {
private:
    std::string json_path;
    std::ofstream json_file;
    std::mutex json_mutex;
    int entries_logged;
    
    void create_directories(const std::string& path) {
        std::filesystem::path p(path);
        std::filesystem::create_directories(p.parent_path());
    }
    
    std::string generate_filename(const std::string& base_path) {
        auto now = std::chrono::system_clock::now();
        auto time_t_now = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << base_path << "/faces_" 
           << std::put_time(std::localtime(&time_t_now), "%Y%m%d_%H%M%S")
           << ".json";
        return ss.str();
    }

public:
    FaceLogger(const std::string& base_path = "logs/faces")
        : entries_logged(0)
    {
        create_directories(base_path);
        json_path = generate_filename(base_path);
        
        json_file.open(json_path, std::ios::out);
        if (!json_file.is_open()) {
            throw std::runtime_error("No se pudo crear archivo JSON");
        }
        
        // Iniciar JSON array
        json_file << "[\n";
        
        spdlog::info("📝 FaceLogger inicializado: {}", json_path);
    }
    
    ~FaceLogger() {
        close();
    }
    
    // Guardar entrada
    bool log_entry(FaceLogEntry& entry) {
        // Auto-generar ID si no existe
        if (entry.id.empty()) {
            entry.id = FaceLogEntry::generate_id();
        }
        
        // Auto-generar timestamp si no existe
        if (entry.timestamp.empty()) {
            entry.timestamp = FaceLogEntry::current_timestamp();
        }
        
        // JSON
        {
            std::lock_guard<std::mutex> lock(json_mutex);
            if (json_file.is_open()) {
                if (entries_logged > 0) {
                    json_file << ",\n";
                }
                json_file << entry.to_json();
                json_file.flush();
            }
        }
        
        entries_logged++;
        
        if (entries_logged % 10 == 0) {
            spdlog::info("📝 {} rostros guardados", entries_logged);
        }
        
        return true;
    }
    
    // Guardar múltiples entradas
    bool log_batch(const std::vector<FaceLogEntry>& entries) {
        for (auto entry : entries) {
            log_entry(entry);
        }
        return true;
    }
    
    // Cerrar archivo
    void close() {
        if (json_file.is_open()) {
            json_file << "\n]\n";
            json_file.close();
            spdlog::info("📝 FaceLogger cerrado. Total: {} rostros", entries_logged);
            spdlog::info("   Archivo: {}", json_path);
        }
    }
    
    // Getters
    int get_entries_count() const { return entries_logged; }
    std::string get_json_path() const { return json_path; }
};