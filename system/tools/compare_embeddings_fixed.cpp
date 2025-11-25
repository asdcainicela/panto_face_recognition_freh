// ============= compare_embeddings_fixed.cpp =============
/*
 * Versión mejorada con 3 estrategias de agrupación:
 * 
 * 1. AVERAGE LINKAGE (recomendado): Compara con el promedio del grupo
 * 2. COMPLETE LINKAGE (estricto): Debe ser similar a TODAS las caras del grupo
 * 3. SINGLE LINKAGE (permisivo): Original, solo primera cara
 * 
 * COMPILAR:
 *   g++ -std=c++17 compare_embeddings_fixed.cpp -o compare_fixed -lsqlite3 -O3
  ./build/bin/compare_embeddings_fixed  logs/faces/freh.db --method complete --threshold 0.90

 * USO:
 *   ./compare_fixed freh.db --method average --threshold 0.60
 *   ./compare_fixed freh.db --method complete --threshold 0.70
 *   ./compare_fixed freh.db --method single --threshold 0.60
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
#include <cstring>
#include <numeric>

enum LinkageMethod {
    SINGLE,      // Original: solo compara con primera cara
    COMPLETE,    // Estricto: debe ser similar a TODAS las caras
    AVERAGE      // Balance: compara con promedio del grupo
};

struct FaceData {
    std::string id;
    std::string timestamp;
    int track_id;
    int age;
    std::string gender;
    std::string emotion;
    std::vector<float> embedding;
    float confidence;
    int person_id = -1;
};

class EmbeddingComparator {
private:
    sqlite3* db;
    float threshold;
    bool verbose;
    LinkageMethod method;
    
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
    
    // ✅ Distancia Euclidiana (alternativa)
    float euclidean_distance(const std::vector<float>& a, const std::vector<float>& b) {
        if (a.size() != b.size() || a.empty()) return 1e9f;
        
        float sum = 0.0f;
        for (size_t i = 0; i < a.size(); i++) {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }
    
    // ✅ NUEVO: Calcular embedding promedio de un grupo
    std::vector<float> calculate_average_embedding(const std::vector<FaceData>& group) {
        if (group.empty()) return {};
        
        size_t dim = group[0].embedding.size();
        std::vector<float> avg(dim, 0.0f);
        
        for (const auto& face : group) {
            for (size_t i = 0; i < dim; i++) {
                avg[i] += face.embedding[i];
            }
        }
        
        for (size_t i = 0; i < dim; i++) {
            avg[i] /= group.size();
        }
        
        // Normalizar (importante para similitud coseno)
        float norm = 0.0f;
        for (float v : avg) norm += v * v;
        norm = std::sqrt(norm);
        
        if (norm > 1e-9) {
            for (float& v : avg) v /= norm;
        }
        
        return avg;
    }
    
    // ✅ NUEVO: Verificar si una cara pertenece a un grupo
    bool belongs_to_group(const FaceData& face, const std::vector<FaceData>& group) {
        if (group.empty()) return true;
        
        switch (method) {
            case SINGLE: {
                // Original: solo compara con primera cara
                float sim = cosine_similarity(face.embedding, group[0].embedding);
                return sim >= threshold;
            }
            
            case COMPLETE: {
                // Estricto: debe ser similar a TODAS las caras del grupo
                for (const auto& member : group) {
                    float sim = cosine_similarity(face.embedding, member.embedding);
                    if (sim < threshold) {
                        if (verbose) {
                            std::cout << "      ✗ Rechazado: sim con " << member.id 
                                     << " = " << std::fixed << std::setprecision(3) << sim << "\n";
                        }
                        return false;
                    }
                }
                return true;
            }
            
            case AVERAGE: {
                // Balance: compara con embedding promedio del grupo
                auto avg_embedding = calculate_average_embedding(group);
                float sim = cosine_similarity(face.embedding, avg_embedding);
                
                if (verbose && sim >= threshold) {
                    std::cout << "      ✓ Agregado: sim con promedio = " 
                             << std::fixed << std::setprecision(3) << sim << "\n";
                }
                
                return sim >= threshold;
            }
        }
        
        return false;
    }

public:
    EmbeddingComparator(const std::string& db_path, float sim_threshold = 0.70f, 
                       LinkageMethod link_method = AVERAGE, bool verb = false)
        : db(nullptr), threshold(sim_threshold), verbose(verb), method(link_method)
    {
        int rc = sqlite3_open_v2(db_path.c_str(), &db, SQLITE_OPEN_READONLY, nullptr);
        if (rc != SQLITE_OK) {
            throw std::runtime_error("❌ No se pudo abrir: " + db_path);
        }
        
        std::cout << "✅ Base de datos abierta: " << db_path << "\n";
        std::cout << "🎯 Threshold: " << std::fixed << std::setprecision(2) << threshold << "\n";
        std::cout << "🔗 Método: ";
        switch (method) {
            case SINGLE:   std::cout << "SINGLE LINKAGE (permisivo)\n"; break;
            case COMPLETE: std::cout << "COMPLETE LINKAGE (estricto)\n"; break;
            case AVERAGE:  std::cout << "AVERAGE LINKAGE (balance) ✅\n"; break;
        }
        std::cout << "\n";
    }
    
    ~EmbeddingComparator() {
        if (db) sqlite3_close(db);
    }
    
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
            throw std::runtime_error("❌ Error en query");
        }
        
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            FaceData face;
            face.id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
            face.timestamp = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
            face.track_id = sqlite3_column_int(stmt, 2);
            face.age = sqlite3_column_int(stmt, 3);
            face.gender = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4));
            face.emotion = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 5));
            
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
    
    // ✅ ALGORITMO MEJORADO
    std::map<int, std::vector<FaceData>> group_by_similarity(std::vector<FaceData>& faces) {
        std::map<int, std::vector<FaceData>> persons;
        int next_person_id = 1;
        
        auto start_time = std::chrono::steady_clock::now();
        
        std::cout << "🔍 Comparando " << faces.size() << " embeddings...\n";
        std::cout << "   (Esto puede tardar varios minutos)\n\n";
        
        size_t total_comparisons = 0;
        
        for (size_t i = 0; i < faces.size(); i++) {
            if (faces[i].person_id != -1) continue;
            
            // Buscar en grupos existentes
            bool found_group = false;
            
            for (auto& [pid, group] : persons) {
                if (belongs_to_group(faces[i], group)) {
                    faces[i].person_id = pid;
                    group.push_back(faces[i]);
                    found_group = true;
                    
                    if (verbose) {
                        std::cout << "   ✓ " << faces[i].id << " → Persona " << pid 
                                 << " (grupo tamaño: " << group.size() << ")\n";
                    }
                    break;
                }
                total_comparisons++;
            }
            
            // Si no pertenece a ningún grupo, crear nuevo
            if (!found_group) {
                faces[i].person_id = next_person_id;
                persons[next_person_id].push_back(faces[i]);
                
                if (verbose) {
                    std::cout << "   ⭐ " << faces[i].id << " → Nueva Persona " 
                             << next_person_id << "\n";
                }
                
                next_person_id++;
            }
            
            // Progress
            if ((i + 1) % 50 == 0) {
                std::cout << "   Procesados: " << (i + 1) << "/" << faces.size() 
                         << " (" << (100 * (i + 1) / faces.size()) << "%)"
                         << " | Personas: " << persons.size() << "\n";
            }
        }
        
        auto end_time = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
        
        std::cout << "\n✅ Comparación completada en " << duration.count() << " segundos\n";
        std::cout << "📊 Total comparaciones: " << total_comparisons << "\n";
        std::cout << "👥 Personas únicas: " << persons.size() << "\n\n";
        
        return persons;
    }
    
    void print_statistics(const std::map<int, std::vector<FaceData>>& persons) {
        std::cout << "\n═══════════════════════════════════════════════════════════════\n";
        std::cout << "                    PERSONAS ÚNICAS IDENTIFICADAS\n";
        std::cout << "═══════════════════════════════════════════════════════════════\n\n";
        
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
            
            std::string most_gender = std::max_element(gender_count.begin(), gender_count.end(),
                [](const auto& a, const auto& b) { return a.second < b.second; })->first;
            
            std::string most_emotion = emotion_count.empty() ? "?" :
                std::max_element(emotion_count.begin(), emotion_count.end(),
                [](const auto& a, const auto& b) { return a.second < b.second; })->first;
            
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
        int total = std::accumulate(sorted_persons.begin(), sorted_persons.end(), 0,
                     [](int sum, const auto& p) { return sum + p.second; });
        std::cout << "  Total de rostros detectados: " << total << "\n";
        std::cout << "  Personas únicas: " << persons.size() << "\n";
        std::cout << "  Promedio detecciones/persona: " << std::fixed << std::setprecision(1)
                  << (float)total / persons.size() << "\n";
        std::cout << "═══════════════════════════════════════════════════════════════\n\n";
    }
    
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
    std::cout << "  --threshold N          Umbral (0.0-1.0, default: 0.70)\n";
    std::cout << "  --method METHOD        Método de agrupación:\n";
    std::cout << "                           single   - Solo primera cara (permisivo)\n";
    std::cout << "                           complete - Todas las caras (estricto)\n";
    std::cout << "                           average  - Promedio grupo (balance) ✅\n";
    std::cout << "  --verbose              Mostrar detalles\n";
    std::cout << "  --export FILE          Exportar a CSV\n\n";
    std::cout << "EJEMPLOS:\n";
    std::cout << "  " << prog << " freh.db --method average --threshold 0.60\n";
    std::cout << "  " << prog << " freh.db --method complete --threshold 0.70\n\n";
    std::cout << "MÉTODOS EXPLICADOS:\n";
    std::cout << "  SINGLE:   Más permisivo, puede agrupar personas diferentes\n";
    std::cout << "  COMPLETE: Más estricto, puede separar la misma persona\n";
    std::cout << "  AVERAGE:  Balance óptimo (RECOMENDADO) ✅\n\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }
    
    std::string db_path = argv[1];
    float threshold = 0.70f;
    LinkageMethod method = AVERAGE;
    bool verbose = false;
    std::string export_file;
    
    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--threshold" && i + 1 < argc) {
            threshold = std::stof(argv[++i]);
        }
        else if (arg == "--method" && i + 1 < argc) {
            std::string m = argv[++i];
            if (m == "single") method = SINGLE;
            else if (m == "complete") method = COMPLETE;
            else if (m == "average") method = AVERAGE;
        }
        else if (arg == "--verbose") {
            verbose = true;
        }
        else if (arg == "--export" && i + 1 < argc) {
            export_file = argv[++i];
        }
    }
    
    try {
        EmbeddingComparator comparator(db_path, threshold, method, verbose);
        
        auto faces = comparator.load_all_faces();
        
        if (faces.empty()) {
            std::cout << "⚠️  No hay rostros con embeddings\n";
            return 0;
        }
        
        auto persons = comparator.group_by_similarity(faces);
        comparator.print_statistics(persons);
        
        if (!export_file.empty()) {
            comparator.export_results(persons, export_file);
        }
        
    } catch (const std::exception& e) {
        std::cerr << "❌ ERROR: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}