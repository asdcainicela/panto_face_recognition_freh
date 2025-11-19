// ============= include/face_database.hpp =============
/*
 * Face Database - SQLite Backend
 * 
 * CARACTERÍSTICAS:
 * - Almacena embeddings + metadata de rostros
 * - Búsqueda rápida por cosine similarity
 * - Soporte para múltiples embeddings por persona
 * - Persistencia en SQLite
 * 
 * TABLA: faces
 * ├── id (INTEGER PRIMARY KEY)
 * ├── name (TEXT)
 * ├── embedding (BLOB) - 512 floats
 * ├── timestamp (TEXT)
 * └── metadata (TEXT) - JSON opcional
 * 
 * OPERACIONES:
 * - add_person(): Registrar nuevo rostro
 * - recognize(): Buscar rostro más similar
 * - update_person(): Actualizar embedding promedio
 * - delete_person(): Eliminar persona
 * - list_all(): Listar todas las personas
 * 
 * AUTOR: PANTO System
 * FECHA: 2025
 */

#pragma once
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <sqlite3.h>

struct FaceRecord {
    int id;
    std::string name;
    std::vector<float> embedding;
    std::string timestamp;
    std::string metadata;
};

struct RecognitionResult {
    std::string name;
    float similarity;
    int person_id;
    bool recognized;  // true si similarity > threshold
};

class FaceDatabase {
private:
    sqlite3* db;
    std::string db_path;
    int embedding_size;
    float recognition_threshold;
    
    bool init_database();
    bool create_tables();
    
    // Helper para serializar embeddings
    static std::vector<unsigned char> serialize_embedding(const std::vector<float>& emb);
    static std::vector<float> deserialize_embedding(const unsigned char* data, int size);

public:
    FaceDatabase(const std::string& db_path, 
                 int embedding_size = 512,
                 float threshold = 0.6f);
    ~FaceDatabase();
    
    // CRUD Operations
    bool add_person(const std::string& name, 
                   const std::vector<float>& embedding,
                   const std::string& metadata = "");
    
    RecognitionResult recognize(const std::vector<float>& embedding);
    
    bool update_embedding(int person_id, const std::vector<float>& new_embedding);
    
    bool delete_person(int person_id);
    bool delete_person(const std::string& name);
    
    // Query operations
    std::vector<FaceRecord> list_all();
    std::vector<FaceRecord> search_by_name(const std::string& name);
    FaceRecord get_by_id(int person_id);
    
    // Stats
    int count_persons();
    
    // Configuración
    void set_threshold(float threshold) { recognition_threshold = threshold; }
    float get_threshold() const { return recognition_threshold; }
    
    // Verificar si la BD está abierta
    bool is_open() const { return db != nullptr; }
};