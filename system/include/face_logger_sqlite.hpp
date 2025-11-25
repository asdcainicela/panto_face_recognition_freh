// ============= include/face_logger_sqlite.hpp =============
/*
 * Face Data Logger - SQLite Backend
 * 
 * VENTAJAS SOBRE JSON:
 * ✅ Consultas SQL rápidas
 * ✅ Índices para búsqueda eficiente
 * ✅ Transacciones ACID
 * ✅ Compresión automática de datos
 * ✅ Sin problemas de concurrencia
 * 
 * SCHEMA:
 * CREATE TABLE faces (
 *   id TEXT PRIMARY KEY,              -- ID único (ej: "a7f3k9m2")
 *   timestamp TEXT NOT NULL,          -- "2025-11-24 14:30:52.123"
 *   age INTEGER,                      -- Edad (25, 30, etc.)
 *   gender TEXT,                      -- "Male" o "Female"
 *   company TEXT DEFAULT 'Freh',      -- Empresa
 *   emotion TEXT,                     -- "Happy", "Sad", etc.
 *   embedding BLOB,                   -- 512 floats (2048 bytes)
 *   track_id INTEGER,                 -- ID del track
 *   confidence REAL,                  -- Confianza de detección
 *   bbox TEXT,                        -- Bounding box (JSON: {"x":10,"y":20,"w":100,"h":120})
 *   created_at INTEGER DEFAULT (strftime('%s', 'now'))
 * );
 * 
 * CREATE INDEX idx_timestamp ON faces(timestamp);
 * CREATE INDEX idx_track_id ON faces(track_id);
 * CREATE INDEX idx_gender ON faces(gender);
 * CREATE INDEX idx_age ON faces(age);
 * 
 * AUTOR: PANTO System
 * FECHA: 2025
 */

#pragma once
#include <string>
#include <vector>
#include <mutex>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <random>
#include <sqlite3.h>
#include <spdlog/spdlog.h>

struct FaceLogEntry {
    std::string id;                      // ID único
    std::string timestamp;               // Timestamp
    int age;                             // Edad
    std::string gender;                  // "Male" o "Female"
    std::string company;                 // "Freh"
    std::string emotion;                 // Emoción
    std::vector<float> embedding;        // 512 valores
    
    // Metadata adicional
    int track_id;                        // ID del track
    float confidence;                    // Confianza de detección
    std::string bbox;                    // Bounding box JSON
    
    FaceLogEntry() 
        : id(""), age(0), gender("Unknown"), 
          company("Freh"), emotion("Unknown"),
          track_id(-1), confidence(0.0f), bbox("") {}
    
    // Generar ID aleatorio (alfanumérico lowercase)
    static std::string generate_id(int length = 8) {
        static const char chars[] = "0123456789abcdefghijklmnopqrstuvwxyz";
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
    
    // Serializar bbox a JSON simple
    static std::string bbox_to_json(int x, int y, int w, int h) {
        std::stringstream ss;
        ss << "{\"x\":" << x << ",\"y\":" << y 
           << ",\"w\":" << w << ",\"h\":" << h << "}";
        return ss.str();
    }
};

class FaceLoggerSQLite {
private:
    sqlite3* db;
    std::string db_path;
    std::mutex db_mutex;
    int entries_logged;
    
    // Prepared statements para mejor performance
    sqlite3_stmt* insert_stmt;
    
    void create_directories(const std::string& path) {
        std::filesystem::path p(path);
        std::filesystem::create_directories(p.parent_path());
    }
    
    std::string generate_db_filename(const std::string& base_path) {
        auto now = std::chrono::system_clock::now();
        auto time_t_now = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << base_path << "/faces_" 
           << std::put_time(std::localtime(&time_t_now), "%Y%m%d_%H%M%S")
           << ".db";
        return ss.str();
    }
    
    bool create_table() {
        const char* sql = R"(
            CREATE TABLE IF NOT EXISTS faces (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                age INTEGER,
                gender TEXT,
                company TEXT DEFAULT 'Freh',
                emotion TEXT,
                embedding BLOB,
                track_id INTEGER,
                confidence REAL,
                bbox TEXT,
                created_at INTEGER DEFAULT (strftime('%s', 'now'))
            );
            
            CREATE INDEX IF NOT EXISTS idx_timestamp ON faces(timestamp);
            CREATE INDEX IF NOT EXISTS idx_track_id ON faces(track_id);
            CREATE INDEX IF NOT EXISTS idx_gender ON faces(gender);
            CREATE INDEX IF NOT EXISTS idx_age ON faces(age);
            CREATE INDEX IF NOT EXISTS idx_created_at ON faces(created_at);
        )";
        
        char* err_msg = nullptr;
        int rc = sqlite3_exec(db, sql, nullptr, nullptr, &err_msg);
        
        if (rc != SQLITE_OK) {
            spdlog::error("❌ Error creando tabla: {}", err_msg);
            sqlite3_free(err_msg);
            return false;
        }
        
        return true;
    }
    
    bool prepare_statements() {
        const char* sql = R"(
            INSERT INTO faces (id, timestamp, age, gender, company, emotion, 
                             embedding, track_id, confidence, bbox)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        )";
        
        int rc = sqlite3_prepare_v2(db, sql, -1, &insert_stmt, nullptr);
        if (rc != SQLITE_OK) {
            spdlog::error("❌ Error preparando statement: {}", sqlite3_errmsg(db));
            return false;
        }
        
        return true;
    }
    
    // Serializar embedding a BLOB
    std::vector<unsigned char> serialize_embedding(const std::vector<float>& emb) {
        std::vector<unsigned char> blob(emb.size() * sizeof(float));
        std::memcpy(blob.data(), emb.data(), blob.size());
        return blob;
    }

public:
    FaceLoggerSQLite(const std::string& base_path = "logs/faces", 
                     const std::string& db_name = "")
        : db(nullptr), entries_logged(0), insert_stmt(nullptr)
    {
        create_directories(base_path);
        
        // Usar nombre custom o generar uno con timestamp
        if (!db_name.empty()) {
            db_path = base_path + "/" + db_name;
        } else {
            db_path = generate_db_filename(base_path);
        }
        
        // Abrir database
        int rc = sqlite3_open(db_path.c_str(), &db);
        if (rc != SQLITE_OK) {
            throw std::runtime_error("No se pudo crear database SQLite: " + std::string(sqlite3_errmsg(db)));
        }
        
        // Optimizaciones de SQLite para mejor performance
        sqlite3_exec(db, "PRAGMA journal_mode=WAL", nullptr, nullptr, nullptr);      // Write-Ahead Logging
        sqlite3_exec(db, "PRAGMA synchronous=NORMAL", nullptr, nullptr, nullptr);    // Balance seguridad/velocidad
        sqlite3_exec(db, "PRAGMA cache_size=10000", nullptr, nullptr, nullptr);      // Cache más grande
        sqlite3_exec(db, "PRAGMA temp_store=MEMORY", nullptr, nullptr, nullptr);     // Temp tables en RAM
        
        // Crear tabla e índices
        if (!create_table()) {
            throw std::runtime_error("No se pudo crear tabla");
        }
        
        // Preparar statements
        if (!prepare_statements()) {
            throw std::runtime_error("No se pudieron preparar statements");
        }
        
        spdlog::info("📝 FaceLoggerSQLite inicializado");
        spdlog::info("   Database: {}", db_path);
        spdlog::info("   Optimizaciones: WAL mode, cache 10K páginas");
    }
    
    ~FaceLoggerSQLite() {
        close();
    }
    
    // Guardar entrada (con transacción implícita)
    bool log_entry(FaceLogEntry& entry) {
        std::lock_guard<std::mutex> lock(db_mutex);
        
        // Auto-generar ID si no existe
        if (entry.id.empty()) {
            entry.id = FaceLogEntry::generate_id();
        }
        
        // Auto-generar timestamp si no existe
        if (entry.timestamp.empty()) {
            entry.timestamp = FaceLogEntry::current_timestamp();
        }
        
        // Serializar embedding
        auto blob = serialize_embedding(entry.embedding);
        
        // Bind parameters
        sqlite3_reset(insert_stmt);
        sqlite3_bind_text(insert_stmt, 1, entry.id.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(insert_stmt, 2, entry.timestamp.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_int(insert_stmt, 3, entry.age);
        sqlite3_bind_text(insert_stmt, 4, entry.gender.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(insert_stmt, 5, entry.company.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(insert_stmt, 6, entry.emotion.c_str(), -1, SQLITE_TRANSIENT);
        
        if (!blob.empty()) {
            sqlite3_bind_blob(insert_stmt, 7, blob.data(), blob.size(), SQLITE_TRANSIENT);
        } else {
            sqlite3_bind_null(insert_stmt, 7);
        }
        
        sqlite3_bind_int(insert_stmt, 8, entry.track_id);
        sqlite3_bind_double(insert_stmt, 9, entry.confidence);
        sqlite3_bind_text(insert_stmt, 10, entry.bbox.c_str(), -1, SQLITE_TRANSIENT);
        
        // Execute
        int rc = sqlite3_step(insert_stmt);
        
        if (rc != SQLITE_DONE) {
            spdlog::error("❌ Error insertando: {}", sqlite3_errmsg(db));
            return false;
        }
        
        entries_logged++;
        
        //if (entries_logged % 10 == 0) {
        //    spdlog::info("📝 {} rostros guardados", entries_logged);
        //}
        
        return true;
    }
    
    // Guardar múltiples entradas en una transacción (MUCHO MÁS RÁPIDO)
    bool log_batch(std::vector<FaceLogEntry>& entries) {
        if (entries.empty()) return true;
        
        std::lock_guard<std::mutex> lock(db_mutex);
        
        // Begin transaction
        sqlite3_exec(db, "BEGIN TRANSACTION", nullptr, nullptr, nullptr);
        
        for (auto& entry : entries) {
            if (entry.id.empty()) {
                entry.id = FaceLogEntry::generate_id();
            }
            if (entry.timestamp.empty()) {
                entry.timestamp = FaceLogEntry::current_timestamp();
            }
            
            auto blob = serialize_embedding(entry.embedding);
            
            sqlite3_reset(insert_stmt);
            sqlite3_bind_text(insert_stmt, 1, entry.id.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(insert_stmt, 2, entry.timestamp.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_bind_int(insert_stmt, 3, entry.age);
            sqlite3_bind_text(insert_stmt, 4, entry.gender.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(insert_stmt, 5, entry.company.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(insert_stmt, 6, entry.emotion.c_str(), -1, SQLITE_TRANSIENT);
            
            if (!blob.empty()) {
                sqlite3_bind_blob(insert_stmt, 7, blob.data(), blob.size(), SQLITE_TRANSIENT);
            } else {
                sqlite3_bind_null(insert_stmt, 7);
            }
            
            sqlite3_bind_int(insert_stmt, 8, entry.track_id);
            sqlite3_bind_double(insert_stmt, 9, entry.confidence);
            sqlite3_bind_text(insert_stmt, 10, entry.bbox.c_str(), -1, SQLITE_TRANSIENT);
            
            if (sqlite3_step(insert_stmt) != SQLITE_DONE) {
                sqlite3_exec(db, "ROLLBACK", nullptr, nullptr, nullptr);
                spdlog::error("❌ Error en batch insert");
                return false;
            }
            
            entries_logged++;
        }
        
        // Commit transaction
        sqlite3_exec(db, "COMMIT", nullptr, nullptr, nullptr);
        
        spdlog::info("📝 Batch guardado: {} rostros (total: {})", 
                    entries.size(), entries_logged);
        
        return true;
    }
    
    // Estadísticas rápidas
    int count_total() {
        std::lock_guard<std::mutex> lock(db_mutex);
        
        const char* sql = "SELECT COUNT(*) FROM faces";
        sqlite3_stmt* stmt;
        
        int count = 0;
        if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) == SQLITE_OK) {
            if (sqlite3_step(stmt) == SQLITE_ROW) {
                count = sqlite3_column_int(stmt, 0);
            }
            sqlite3_finalize(stmt);
        }
        
        return count;
    }
    
    // Estadísticas por género
    void print_statistics() {
        std::lock_guard<std::mutex> lock(db_mutex);
        
        const char* sql = R"(
            SELECT 
                COUNT(*) as total,
                COUNT(DISTINCT track_id) as unique_tracks,
                AVG(age) as avg_age,
                SUM(CASE WHEN gender='Male' THEN 1 ELSE 0 END) as males,
                SUM(CASE WHEN gender='Female' THEN 1 ELSE 0 END) as females,
                AVG(confidence) as avg_confidence
            FROM faces
        )";
        
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) == SQLITE_OK) {
            if (sqlite3_step(stmt) == SQLITE_ROW) {
                int total = sqlite3_column_int(stmt, 0);
                int tracks = sqlite3_column_int(stmt, 1);
                double avg_age = sqlite3_column_double(stmt, 2);
                int males = sqlite3_column_int(stmt, 3);
                int females = sqlite3_column_int(stmt, 4);
                double avg_conf = sqlite3_column_double(stmt, 5);
                
                /*spdlog::info("📊 ESTADÍSTICAS:");
                spdlog::info("   Total rostros: {}", total);
                spdlog::info("   Tracks únicos: {}", tracks);
                spdlog::info("   Edad promedio: {:.1f} años", avg_age);
                spdlog::info("   Género: {} Hombres, {} Mujeres", males, females);
                spdlog::info("   Confianza promedio: {:.2f}", avg_conf);*/
            }
            sqlite3_finalize(stmt);
        }
    }
    
    // Cerrar database
    void close() {
        if (insert_stmt) {
            sqlite3_finalize(insert_stmt);
            insert_stmt = nullptr;
        }
        
        if (db) {
            // Optimizar antes de cerrar
            sqlite3_exec(db, "VACUUM", nullptr, nullptr, nullptr);
            sqlite3_close(db);
            db = nullptr;
            
            spdlog::info(" FaceLoggerSQLite cerrado");
            //spdlog::info("   Total: {} rostros", entries_logged);
            spdlog::info("   Database: {}", db_path);
            
            // Mostrar tamaño del archivo
            try {
                auto size = std::filesystem::file_size(db_path);
                spdlog::info("   Tamaño: {:.2f} MB", size / 1024.0 / 1024.0);
            } catch (...) {}
        }
    }
    
    // Getters
    int get_entries_count() const { return entries_logged; }
    std::string get_db_path() const { return db_path; }
    bool is_open() const { return db != nullptr; }
};