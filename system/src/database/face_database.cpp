// ============= src/database/face_database.cpp =============
#include "database/face_database.hpp"  // ⭐ CAMBIO
#include "recognition/recognizer.hpp"  // ⭐ CAMBIO
#include <spdlog/spdlog.h>
#include <cstring>
#include <ctime>
#include <filesystem>

// ... resto del código

// ==================== CONSTRUCTOR/DESTRUCTOR ====================

FaceDatabase::FaceDatabase(const std::string& db_path, 
                           int embedding_size,
                           float threshold)
    : db(nullptr), db_path(db_path), 
      embedding_size(embedding_size),
      recognition_threshold(threshold)
{
    spdlog::info(" Inicializando Face Database");
    spdlog::info("   Path: {}", db_path);
    spdlog::info("   Embedding size: {}", embedding_size);
    spdlog::info("   Recognition threshold: {:.2f}", threshold);
    
    // Create directory if not exists
    std::filesystem::path p(db_path);
    std::filesystem::create_directories(p.parent_path());
    
    if (!init_database()) {
        throw std::runtime_error("No se pudo inicializar la base de datos");
    }
    
    spdlog::info(" Database ready ({} persons)", count_persons());
}

FaceDatabase::~FaceDatabase() {
    if (db) {
        sqlite3_close(db);
    }
}

// ==================== INITIALIZATION ====================

bool FaceDatabase::init_database() {
    int rc = sqlite3_open(db_path.c_str(), &db);
    if (rc != SQLITE_OK) {
        spdlog::error("Cannot open database: {}", sqlite3_errmsg(db));
        return false;
    }
    
    return create_tables();
}

bool FaceDatabase::create_tables() {
    const char* sql = R"(
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            embedding BLOB NOT NULL,
            timestamp TEXT NOT NULL,
            metadata TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_name ON faces(name);
    )";
    
    char* err_msg = nullptr;
    int rc = sqlite3_exec(db, sql, nullptr, nullptr, &err_msg);
    
    if (rc != SQLITE_OK) {
        spdlog::error("SQL error: {}", err_msg);
        sqlite3_free(err_msg);
        return false;
    }
    
    return true;
}

// ==================== SERIALIZATION ====================

std::vector<unsigned char> FaceDatabase::serialize_embedding(const std::vector<float>& emb) {
    std::vector<unsigned char> blob(emb.size() * sizeof(float));
    std::memcpy(blob.data(), emb.data(), blob.size());
    return blob;
}

std::vector<float> FaceDatabase::deserialize_embedding(const unsigned char* data, int size) {
    std::vector<float> emb(size / sizeof(float));
    std::memcpy(emb.data(), data, size);
    return emb;
}

// ==================== ADD PERSON ====================

bool FaceDatabase::add_person(const std::string& name,
                              const std::vector<float>& embedding,
                              const std::string& metadata)
{
    if (embedding.size() != static_cast<size_t>(embedding_size)) {
        spdlog::error("Invalid embedding size: {} (expected {})", 
                     embedding.size(), embedding_size);
        return false;
    }
    
    // Get current timestamp
    std::time_t now = std::time(nullptr);
    char timestamp[32];
    std::strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", 
                 std::localtime(&now));
    
    // Serialize embedding
    auto blob = serialize_embedding(embedding);
    
    // Insert into database
    const char* sql = "INSERT INTO faces (name, embedding, timestamp, metadata) VALUES (?, ?, ?, ?)";
    sqlite3_stmt* stmt;
    
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        spdlog::error("Failed to prepare statement: {}", sqlite3_errmsg(db));
        return false;
    }
    
    sqlite3_bind_text(stmt, 1, name.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_blob(stmt, 2, blob.data(), blob.size(), SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 3, timestamp, -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 4, metadata.c_str(), -1, SQLITE_TRANSIENT);
    
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    if (rc != SQLITE_DONE) {
        spdlog::error("Failed to insert: {}", sqlite3_errmsg(db));
        return false;
    }
    
    int person_id = sqlite3_last_insert_rowid(db);
    spdlog::info("✓ Added person: {} (ID={})", name, person_id);
    
    return true;
}

// ==================== RECOGNIZE ====================

RecognitionResult FaceDatabase::recognize(const std::vector<float>& embedding) {
    RecognitionResult result;
    result.name = "Unknown";
    result.similarity = 0.0f;
    result.person_id = -1;
    result.recognized = false;
    
    if (embedding.size() != static_cast<size_t>(embedding_size)) {
        spdlog::error("Invalid embedding size for recognition");
        return result;
    }
    
    // Get all faces
    const char* sql = "SELECT id, name, embedding FROM faces";
    sqlite3_stmt* stmt;
    
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        spdlog::error("Failed to prepare query: {}", sqlite3_errmsg(db));
        return result;
    }
    
    float best_similarity = 0.0f;
    int best_id = -1;
    std::string best_name;
    
    // Compare with all stored embeddings
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        int id = sqlite3_column_int(stmt, 0);
        const char* name = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        const unsigned char* blob = static_cast<const unsigned char*>(sqlite3_column_blob(stmt, 2));
        int blob_size = sqlite3_column_bytes(stmt, 2);
        
        auto stored_emb = deserialize_embedding(blob, blob_size);
        float similarity = FaceRecognizer::compare(embedding, stored_emb);
        
        if (similarity > best_similarity) {
            best_similarity = similarity;
            best_id = id;
            best_name = name;
        }
    }
    
    sqlite3_finalize(stmt);
    
    // Check threshold
    if (best_similarity >= recognition_threshold) {
        result.name = best_name;
        result.similarity = best_similarity;
        result.person_id = best_id;
        result.recognized = true;
    }
    
    return result;
}

// ==================== UPDATE ====================

bool FaceDatabase::update_embedding(int person_id, const std::vector<float>& new_embedding) {
    if (new_embedding.size() != static_cast<size_t>(embedding_size)) {
        spdlog::error("Invalid embedding size");
        return false;
    }
    
    auto blob = serialize_embedding(new_embedding);
    
    const char* sql = "UPDATE faces SET embedding = ? WHERE id = ?";
    sqlite3_stmt* stmt;
    
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) return false;
    
    sqlite3_bind_blob(stmt, 1, blob.data(), blob.size(), SQLITE_TRANSIENT);
    sqlite3_bind_int(stmt, 2, person_id);
    
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    return rc == SQLITE_DONE;
}

// ==================== DELETE ====================

bool FaceDatabase::delete_person(int person_id) {
    const char* sql = "DELETE FROM faces WHERE id = ?";
    sqlite3_stmt* stmt;
    
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) return false;
    
    sqlite3_bind_int(stmt, 1, person_id);
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    if (rc == SQLITE_DONE) {
        spdlog::info("✓ Deleted person ID={}", person_id);
        return true;
    }
    return false;
}

bool FaceDatabase::delete_person(const std::string& name) {
    const char* sql = "DELETE FROM faces WHERE name = ?";
    sqlite3_stmt* stmt;
    
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) return false;
    
    sqlite3_bind_text(stmt, 1, name.c_str(), -1, SQLITE_TRANSIENT);
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    if (rc == SQLITE_DONE) {
        spdlog::info("✓ Deleted person: {}", name);
        return true;
    }
    return false;
}

// ==================== QUERY ====================

std::vector<FaceRecord> FaceDatabase::list_all() {
    std::vector<FaceRecord> records;
    
    const char* sql = "SELECT id, name, embedding, timestamp, metadata FROM faces";
    sqlite3_stmt* stmt;
    
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) return records;
    
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        FaceRecord record;
        record.id = sqlite3_column_int(stmt, 0);
        record.name = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        
        const unsigned char* blob = static_cast<const unsigned char*>(sqlite3_column_blob(stmt, 2));
        int blob_size = sqlite3_column_bytes(stmt, 2);
        record.embedding = deserialize_embedding(blob, blob_size);
        
        record.timestamp = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        const char* meta = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4));
        record.metadata = meta ? meta : "";
        
        records.push_back(record);
    }
    
    sqlite3_finalize(stmt);
    return records;
}

std::vector<FaceRecord> FaceDatabase::search_by_name(const std::string& name) {
    std::vector<FaceRecord> records;
    
    const char* sql = "SELECT id, name, embedding, timestamp, metadata FROM faces WHERE name LIKE ?";
    sqlite3_stmt* stmt;
    
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) return records;
    
    std::string pattern = "%" + name + "%";
    sqlite3_bind_text(stmt, 1, pattern.c_str(), -1, SQLITE_TRANSIENT);
    
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        FaceRecord record;
        record.id = sqlite3_column_int(stmt, 0);
        record.name = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        
        const unsigned char* blob = static_cast<const unsigned char*>(sqlite3_column_blob(stmt, 2));
        int blob_size = sqlite3_column_bytes(stmt, 2);
        record.embedding = deserialize_embedding(blob, blob_size);
        
        record.timestamp = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        const char* meta = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4));
        record.metadata = meta ? meta : "";
        
        records.push_back(record);
    }
    
    sqlite3_finalize(stmt);
    return records;
}

FaceRecord FaceDatabase::get_by_id(int person_id) {
    FaceRecord record;
    record.id = -1;
    
    const char* sql = "SELECT id, name, embedding, timestamp, metadata FROM faces WHERE id = ?";
    sqlite3_stmt* stmt;
    
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) return record;
    
    sqlite3_bind_int(stmt, 1, person_id);
    
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        record.id = sqlite3_column_int(stmt, 0);
        record.name = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        
        const unsigned char* blob = static_cast<const unsigned char*>(sqlite3_column_blob(stmt, 2));
        int blob_size = sqlite3_column_bytes(stmt, 2);
        record.embedding = deserialize_embedding(blob, blob_size);
        
        record.timestamp = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        const char* meta = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4));
        record.metadata = meta ? meta : "";
    }
    
    sqlite3_finalize(stmt);
    return record;
}

int FaceDatabase::count_persons() {
    const char* sql = "SELECT COUNT(*) FROM faces";
    sqlite3_stmt* stmt;
    
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) return 0;
    
    int count = 0;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        count = sqlite3_column_int(stmt, 0);
    }
    
    sqlite3_finalize(stmt);
    return count;
}