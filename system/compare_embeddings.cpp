// ============= compare_embeddings.cpp =============
/*
 * Compara embeddings de la base de datos para identificar personas únicas
 * 
 * COMPILAR:
 *   g++ -std=c++17 compare_embeddings.cpp -o compare_embeddings -lsqlite3 -O3
 * 
 * USO:
 *   ./compare_embeddings freh.db --threshold 0.70
 *   ./compare_embeddings freh.db --threshold 0.75 --verbose
 *   ./compare_embeddings freh.db --export unique_persons.txt
 */

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <sqlite3.h>
#include <algorithm>
#include <chrono>

struct FaceData {
    std::string id;           // ID único del registro
    std::string timestamp;    // Fecha/hora
    int track_id;            // ID del track
    int age;                 // Edad
    std::string gender;      // Género
    std::string emotion;     // Emoción
    std::vector<float> embedding;  // ✅ Vector de 512 floats
    float confidence;        // Confianza
    
    int person_id = -1;      // ID de persona única (calculado)
};

class EmbeddingComparator {
private:
    sqlite3* db;
    float threshold;
    bool verbose;
    
    // ✅ Deserializar BLOB a vector de floats
    std::vector<float> blob_to_embedding(const void* blob, int size) {
        if (!blob || size == 0) return {};
        
        int num_floats = size / sizeof(float);
        std::vector<float> embedding(num_floats);
        std::memcpy(embedding.data(), blob, size);
        
        return embedding;
    }
    
    // ✅ Similitud Coseno (0.0 a 1.0)
    float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
        if (a.size() != b.size() || a.empty()) return 0.0f;
        
        float dot = 0.0f;
        float norm_a = 0.0f;
        float norm_b = 0.0f;
        
        for (size_t i = 0; i < a.size(); i++) {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }
        
        norm_a = std::sqrt(norm_a);
        norm_b = std::sqrt(norm_b);
        
        if (norm_a < 1e-9 || norm_b < 1e-9) return 0.0f;
        
        return dot / (norm_a * norm_b);
    }

public:
    EmbeddingComparator(const std::string& db_path, float sim_threshold = 0.70f, bool verb = false)
        : db(nullptr), threshold(sim_threshold), verbose(verb)
    {
        int rc = sqlite3_open_v2(db_path.c_str(), &db, SQLITE_OPEN_READONLY, nullptr);
        if (rc != SQLITE_OK) {
            throw std::runtime_error("❌ No se pudo abrir: " + db_path);
        }
        
        std::cout << "✅ Base de datos abierta: " << db_path << "\n";
        std::cout << "🎯 Threshold de similitud: " << std::fixed << std::setprecision(2) 
                  << threshold << "\n\n";
    }
    
    ~EmbeddingComparator() {
        if (db) sqlite3_close(db);
    }
    
    // ✅ Cargar TODOS los rostros con embeddings
    std::vector<FaceData> load_all_faces() {
        std::vector<FaceData> faces;
        
        const char* sql = R"(
            SELECT id, timestamp, track_id, age, gender, emotion, embedding, confidence
            FROM faces
            WHERE embedding IS NOT NULL
            ORDER BY timestamp ASC
        )";
        
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
            throw std::runtime_error("❌ Error en query: " + std::string(sqlite3_errmsg(db)));
        }
        
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            FaceData face;
            
            face.id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
            face.timestamp = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
            face.track_id = sqlite3_column_int(stmt, 2);
            face.age = sqlite3_column_int(stmt, 3);
            face.gender = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4));
            face.emotion = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 5));
            
            // ✅ LEER EMBEDDING (BLOB)
            const void* blob = sqlite3_column_blob(stmt, 6);
            int blob_size = sqlite3_column_bytes(stmt, 6);
            face.embedding = blob_to_embedding(blob, blob_size);
            
            face.confidence = sqlite3_column_double(stmt, 7);
            
            faces.push_back(face);
        }
        
        sqlite3_finalize(stmt);
        
        std::cout << "📥 Cargados " << faces.size() << " rostros\n";
        std::cout << "📊 Dimensión embedding: " << (faces.empty() ? 0 : faces[0].embedding.size()) << "\n\n";
        
        return faces;
    }
    
    // ✅ Agrupar por similitud de embeddings
    std::map<int, std::vector<FaceData>> group_by_similarity(std::vector<FaceData>& faces) {
        std::map<int, std::vector<FaceData>> persons;
        int next_person_id = 1;
        
        auto start_time = std::chrono::steady_clock::now();
        
        std::cout << "🔍 Comparando " << faces.size() << " embeddings...\n";
        std::cout << "   (Esto puede tardar varios minutos)\n\n";
        
        size_t total_comparisons = 0;
        
        for (size_t i = 0; i < faces.size(); i++) {
            // Ya fue asignado a una persona
            if (faces[i].person_id != -1) continue;
            
            // Crear nuevo grupo de persona
            faces[i].person_id = next_person_id;
            persons[next_person_id].push_back(faces[i]);
            
            // Comparar con rostros posteriores
            for (size_t j = i + 1; j < faces.size(); j++) {
                if (faces[j].person_id != -1) continue;
                
                float similarity = cosine_similarity(faces[i].embedding, faces[j].embedding);
                total_comparisons++;
                
                // ✅ Si supera el threshold, es la MISMA persona
                if (similarity >= threshold) {
                    faces[j].person_id = next_person_id;
                    persons[next_person_id].push_back(faces[j]);
                    
                    if (verbose) {
                        std::cout << "   ✓ Match: " << faces[i].id << " ↔ " << faces[j].id 
                                  << " (sim=" << std::fixed << std::setprecision(3) << similarity << ")\n";
                    }
                }
            }
            
            next_person_id++;
            
            // Progress
            if ((i + 1) % 50 == 0) {
                std::cout << "   Procesados: " << (i + 1) << "/" << faces.size() 
                          << " (" << (100 * (i + 1) / faces.size()) << "%)\n";
            }
        }
        
        auto end_time = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
        
        std::cout << "\n✅ Comparación completada en " << duration.count() << " segundos\n";
        std::cout << "📊 Total comparaciones: " << total_comparisons << "\n";
        std::cout << "👥 Personas únicas: " << persons.size() << "\n\n";
        
        return persons;
    }
    
    // ✅ Mostrar estadísticas por persona
    void print_statistics(const std::map<int, std::vector<FaceData>>& persons) {
        std::cout << "\n═══════════════════════════════════════════════════════════════\n";
        std::cout << "                    PERSONAS ÚNICAS IDENTIFICADAS\n";
        std::cout << "═══════════════════════════════════════════════════════════════\n\n";
        
        // Ordenar por cantidad de detecciones (descendente)
        std::vector<std::pair<int, int>> sorted_persons;
        for (const auto& [pid, records] : persons) {
            sorted_persons.push_back({pid, records.size()});
        }
        std::sort(sorted_persons.begin(), sorted_persons.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        
        std::cout << std::setw(8) << "Person"
                  << std::setw(12) << "Detecciones"
                  << std::setw(6) << "Edad"
                  << std::setw(10) << "Género"
                  << std::setw(12) << "Emoción"
                  << std::setw(12) << "Tracks"
                  << "\n";
        std::cout << std::string(70, '-') << "\n";
        
        for (const auto& [person_id, count] : sorted_persons) {
            const auto& records = persons.at(person_id);
            
            // Calcular valores representativos
            int sum_age = 0;
            std::map<std::string, int> gender_count;
            std::map<std::string, int> emotion_count;
            std::set<int> unique_tracks;
            
            for (const auto& r : records) {
                sum_age += r.age;
                gender_count[r.gender]++;
                if (r.emotion != "Unknown") emotion_count[r.emotion]++;
                unique_tracks.insert(r.track_id);
            }
            
            int avg_age = sum_age / records.size();
            
            // Género más frecuente
            std::string most_gender = "?";
            int max_gender = 0;
            for (const auto& [g, c] : gender_count) {
                if (c > max_gender) {
                    max_gender = c;
                    most_gender = g;
                }
            }
            
            // Emoción más frecuente
            std::string most_emotion = "?";
            int max_emotion = 0;
            for (const auto& [e, c] : emotion_count) {
                if (c > max_emotion) {
                    max_emotion = c;
                    most_emotion = e;
                }
            }
            
            std::cout << std::setw(8) << person_id
                      << std::setw(12) << count
                      << std::setw(6) << avg_age
                      << std::setw(10) << most_gender
                      << std::setw(12) << most_emotion
                      << std::setw(12) << unique_tracks.size()
                      << "\n";
        }
        
        std::cout << "\n";
        std::cout << "RESUMEN:\n";
        std::cout << "  Total de rostros detectados: " << std::accumulate(sorted_persons.begin(), sorted_persons.end(), 0,
                     [](int sum, const auto& p) { return sum + p.second; }) << "\n";
        std::cout << "  Personas únicas: " << persons.size() << "\n";
        std::cout << "  Promedio detecciones/persona: " 
                  << std::fixed << std::setprecision(1)
                  << (float)std::accumulate(sorted_persons.begin(), sorted_persons.end(), 0,
                     [](int sum, const auto& p) { return sum + p.second; }) / persons.size() << "\n";
        std::cout << "═══════════════════════════════════════════════════════════════\n\n";
    }
    
    // ✅ Exportar resultados
    void export_results(const std::map<int, std::vector<FaceData>>& persons, const std::string& filename) {
        std::ofstream file(filename);
        if (!file) {
            std::cerr << "❌ No se pudo crear: " << filename << "\n";
            return;
        }
        
        file << "person_id,detections,avg_age,gender,emotion,tracks,first_seen,last_seen\n";
        
        for (const auto& [pid, records] : persons) {
            int sum_age = 0;
            std::map<std::string, int> gender_count;
            std::map<std::string, int> emotion_count;
            std::set<int> tracks;
            
            for (const auto& r : records) {
                sum_age += r.age;
                gender_count[r.gender]++;
                emotion_count[r.emotion]++;
                tracks.insert(r.track_id);
            }
            
            std::string most_gender = std::max_element(gender_count.begin(), gender_count.end(),
                [](const auto& a, const auto& b) { return a.second < b.second; })->first;
            
            std::string most_emotion = emotion_count.empty() ? "Unknown" :
                std::max_element(emotion_count.begin(), emotion_count.end(),
                [](const auto& a, const auto& b) { return a.second < b.second; })->first;
            
            file << pid << ","
                 << records.size() << ","
                 << (sum_age / records.size()) << ","
                 << most_gender << ","
                 << most_emotion << ","
                 << tracks.size() << ","
                 << records.front().timestamp << ","
                 << records.back().timestamp << "\n";
        }
        
        file.close();
        std::cout << "💾 Exportado a: " << filename << "\n\n";
    }
};

void print_usage(const char* prog) {
    std::cout << "USO: " << prog << " <database.db> [opciones]\n\n";
    std::cout << "OPCIONES:\n";
    std::cout << "  --threshold N      Umbral de similitud (0.0-1.0, default: 0.70)\n";
    std::cout << "  --verbose          Mostrar cada match encontrado\n";
    std::cout << "  --export FILE      Exportar resultados a CSV\n\n";
    std::cout << "EJEMPLOS:\n";
    std::cout << "  " << prog << " freh.db\n";
    std::cout << "  " << prog << " freh.db --threshold 0.75\n";
    std::cout << "  " << prog << " freh.db --threshold 0.70 --export results.csv\n\n";
    std::cout << "THRESHOLDS RECOMENDADOS:\n";
    std::cout << "  0.60-0.65: Permisivo (puede juntar personas diferentes)\n";
    std::cout << "  0.70-0.75: Balance óptimo ✅\n";
    std::cout << "  0.80+:     Estricto (puede separar la misma persona)\n\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }
    
    std::string db_path = argv[1];
    float threshold = 0.70f;
    bool verbose = false;
    std::string export_file;
    
    // Parsear argumentos
    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--threshold" && i + 1 < argc) {
            threshold = std::stof(argv[++i]);
        }
        else if (arg == "--verbose") {
            verbose = true;
        }
        else if (arg == "--export" && i + 1 < argc) {
            export_file = argv[++i];
        }
    }
    
    try {
        EmbeddingComparator comparator(db_path, threshold, verbose);
        
        // Cargar rostros
        auto faces = comparator.load_all_faces();
        
        if (faces.empty()) {
            std::cout << "⚠️  No hay rostros con embeddings en la base de datos\n";
            return 0;
        }
        
        // Agrupar por similitud
        auto persons = comparator.group_by_similarity(faces);
        
        // Mostrar estadísticas
        comparator.print_statistics(persons);
        
        // Exportar si se solicitó
        if (!export_file.empty()) {
            comparator.export_results(persons, export_file);
        }
        
    } catch (const std::exception& e) {
        std::cerr << "❌ ERROR: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}