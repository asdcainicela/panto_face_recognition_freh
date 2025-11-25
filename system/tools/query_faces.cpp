// ============= tools/query_faces.cpp =============
/*
 * Herramienta de consulta para face_logger SQLite
 * 
 * EJEMPLOS DE USO:
 * 
 *
  ./build/bin/query_faces  logs/faces/freh.db --recent 5

 * ./query_faces faces_20251124_143052.db --stats
 * ./query_faces faces.db --recent 10
 * ./query_faces faces.db --gender Male --age-range 25 35
 * ./query_faces faces.db --emotion Happy --export happy_faces.csv
 * ./query_faces faces.db --track 123
 */

#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include <fstream>
#include <sqlite3.h>
#include <spdlog/spdlog.h>

struct QueryResult {
    std::string id;
    std::string timestamp;
    int age;
    std::string gender;
    std::string emotion;
    int track_id;
    float confidence;
    std::string bbox;
};

class FaceQueryTool {
private:
    sqlite3* db;
    std::string db_path;

public:
    FaceQueryTool(const std::string& path) : db(nullptr), db_path(path) {
        int rc = sqlite3_open(path.c_str(), &db);
        if (rc != SQLITE_OK) {
            throw std::runtime_error("No se pudo abrir database: " + std::string(sqlite3_errmsg(db)));
        }
    }

    ~FaceQueryTool() {
        if (db) sqlite3_close(db);
    }

    // Estadísticas generales
    void show_statistics() {
        const char* sql = R"(
            SELECT 
                COUNT(*) as total,
                COUNT(DISTINCT track_id) as unique_tracks,
                MIN(age) as min_age,
                MAX(age) as max_age,
                AVG(age) as avg_age,
                SUM(CASE WHEN gender='Male' THEN 1 ELSE 0 END) as males,
                SUM(CASE WHEN gender='Female' THEN 1 ELSE 0 END) as females,
                MIN(timestamp) as first_timestamp,
                MAX(timestamp) as last_timestamp,
                AVG(confidence) as avg_confidence
            FROM faces
        )";

        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
            std::cerr << "Error preparando query: " << sqlite3_errmsg(db) << std::endl;
            return;
        }

        if (sqlite3_step(stmt) == SQLITE_ROW) {
            std::cout << "\n═══════════════════════════════════════════════" << std::endl;
            std::cout << "   ESTADÍSTICAS GENERALES" << std::endl;
            std::cout << "═══════════════════════════════════════════════" << std::endl;
            std::cout << "Total rostros:      " << sqlite3_column_int(stmt, 0) << std::endl;
            std::cout << "Tracks únicos:      " << sqlite3_column_int(stmt, 1) << std::endl;
            std::cout << "Edad (min/max/avg): " << sqlite3_column_int(stmt, 2) << " / "
                      << sqlite3_column_int(stmt, 3) << " / "
                      << std::fixed << std::setprecision(1) << sqlite3_column_double(stmt, 4) << std::endl;
            std::cout << "Género:             " << sqlite3_column_int(stmt, 5) << " Hombres, "
                      << sqlite3_column_int(stmt, 6) << " Mujeres" << std::endl;
            std::cout << "Primera detección:  " << (const char*)sqlite3_column_text(stmt, 7) << std::endl;
            std::cout << "Última detección:   " << (const char*)sqlite3_column_text(stmt, 8) << std::endl;
            std::cout << "Confianza promedio: " << std::fixed << std::setprecision(2)
                      << sqlite3_column_double(stmt, 9) << std::endl;
            std::cout << "═══════════════════════════════════════════════\n" << std::endl;
        }

        sqlite3_finalize(stmt);

        // Distribución por emoción
        show_emotion_distribution();
        
        // Distribución por edad
        show_age_distribution();
    }

    void show_emotion_distribution() {
        const char* sql = R"(
            SELECT emotion, COUNT(*) as count 
            FROM faces 
            WHERE emotion != 'Unknown'
            GROUP BY emotion 
            ORDER BY count DESC
        )";

        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) != SQLITE_OK) return;

        std::cout << "EMOCIONES:" << std::endl;
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            std::cout << "  " << std::setw(12) << std::left << (const char*)sqlite3_column_text(stmt, 0)
                      << ": " << sqlite3_column_int(stmt, 1) << std::endl;
        }
        std::cout << std::endl;

        sqlite3_finalize(stmt);
    }

    void show_age_distribution() {
        const char* sql = R"(
            SELECT 
                CASE 
                    WHEN age < 10 THEN '0-9'
                    WHEN age < 20 THEN '10-19'
                    WHEN age < 30 THEN '20-29'
                    WHEN age < 40 THEN '30-39'
                    WHEN age < 50 THEN '40-49'
                    WHEN age < 60 THEN '50-59'
                    ELSE '60+'
                END as age_group,
                COUNT(*) as count
            FROM faces
            GROUP BY age_group
            ORDER BY age_group
        )";

        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) != SQLITE_OK) return;

        std::cout << "DISTRIBUCIÓN POR EDAD:" << std::endl;
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            std::cout << "  " << std::setw(8) << std::left << (const char*)sqlite3_column_text(stmt, 0)
                      << ": " << sqlite3_column_int(stmt, 1) << std::endl;
        }
        std::cout << std::endl;

        sqlite3_finalize(stmt);
    }

    // Consultas con filtros
    std::vector<QueryResult> query(const std::string& conditions = "", int limit = 100) {
        std::vector<QueryResult> results;

        std::string sql = "SELECT id, timestamp, age, gender, emotion, track_id, confidence, bbox FROM faces";
        if (!conditions.empty()) {
            sql += " WHERE " + conditions;
        }
        sql += " ORDER BY timestamp DESC LIMIT " + std::to_string(limit);

        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
            std::cerr << "Error: " << sqlite3_errmsg(db) << std::endl;
            return results;
        }

        while (sqlite3_step(stmt) == SQLITE_ROW) {
            QueryResult r;
            r.id = (const char*)sqlite3_column_text(stmt, 0);
            r.timestamp = (const char*)sqlite3_column_text(stmt, 1);
            r.age = sqlite3_column_int(stmt, 2);
            r.gender = (const char*)sqlite3_column_text(stmt, 3);
            r.emotion = (const char*)sqlite3_column_text(stmt, 4);
            r.track_id = sqlite3_column_int(stmt, 5);
            r.confidence = sqlite3_column_double(stmt, 6);
            
            const char* bbox_str = (const char*)sqlite3_column_text(stmt, 7);
            r.bbox = bbox_str ? bbox_str : "";

            results.push_back(r);
        }

        sqlite3_finalize(stmt);
        return results;
    }

    void print_results(const std::vector<QueryResult>& results) {
        if (results.empty()) {
            std::cout << "No se encontraron resultados." << std::endl;
            return;
        }

        std::cout << "\nEncontrados " << results.size() << " rostros:\n" << std::endl;
        std::cout << std::setw(12) << "ID" 
                  << std::setw(22) << "Timestamp"
                  << std::setw(5) << "Age"
                  << std::setw(8) << "Gender"
                  << std::setw(12) << "Emotion"
                  << std::setw(8) << "Track"
                  << std::setw(8) << "Conf"
                  << std::endl;
        std::cout << std::string(80, '-') << std::endl;

        for (const auto& r : results) {
            std::cout << std::setw(12) << r.id
                      << std::setw(22) << r.timestamp
                      << std::setw(5) << r.age
                      << std::setw(8) << r.gender
                      << std::setw(12) << r.emotion
                      << std::setw(8) << r.track_id
                      << std::setw(8) << std::fixed << std::setprecision(2) << r.confidence
                      << std::endl;
        }
    }

    void export_csv(const std::vector<QueryResult>& results, const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error abriendo archivo: " << filename << std::endl;
            return;
        }

        // Header
        file << "id,timestamp,age,gender,emotion,track_id,confidence,bbox\n";

        // Data
        for (const auto& r : results) {
            file << r.id << ","
                 << r.timestamp << ","
                 << r.age << ","
                 << r.gender << ","
                 << r.emotion << ","
                 << r.track_id << ","
                 << std::fixed << std::setprecision(3) << r.confidence << ","
                 << "\"" << r.bbox << "\"\n";
        }

        file.close();
        std::cout << "Exportado a: " << filename << std::endl;
    }
};

void print_usage(const char* prog) {
    std::cout << "USO: " << prog << " <database.db> [opciones]\n\n";
    std::cout << "OPCIONES:\n";
    std::cout << "  --stats                     Mostrar estadísticas generales\n";
    std::cout << "  --recent N                  Mostrar últimos N rostros\n";
    std::cout << "  --gender [Male|Female]      Filtrar por género\n";
    std::cout << "  --age-range MIN MAX         Filtrar por rango de edad\n";
    std::cout << "  --emotion EMOTION           Filtrar por emoción\n";
    std::cout << "  --track ID                  Filtrar por track ID\n";
    std::cout << "  --export FILENAME.csv       Exportar resultados a CSV\n";
    std::cout << "\nEJEMPLOS:\n";
    std::cout << "  " << prog << " faces.db --stats\n";
    std::cout << "  " << prog << " faces.db --recent 20\n";
    std::cout << "  " << prog << " faces.db --gender Male --age-range 25 35\n";
    std::cout << "  " << prog << " faces.db --emotion Happy --export happy.csv\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    try {
        FaceQueryTool tool(argv[1]);

        bool stats_only = false;
        std::string conditions;
        int limit = 100;
        std::string export_file;

        for (int i = 2; i < argc; i++) {
            std::string arg = argv[i];

            if (arg == "--stats") {
                stats_only = true;
            }
            else if (arg == "--recent" && i + 1 < argc) {
                limit = std::stoi(argv[++i]);
            }
            else if (arg == "--gender" && i + 1 < argc) {
                std::string gender = argv[++i];
                if (!conditions.empty()) conditions += " AND ";
                conditions += "gender='" + gender + "'";
            }
            else if (arg == "--age-range" && i + 2 < argc) {
                int min_age = std::stoi(argv[++i]);
                int max_age = std::stoi(argv[++i]);
                if (!conditions.empty()) conditions += " AND ";
                conditions += "age BETWEEN " + std::to_string(min_age) + " AND " + std::to_string(max_age);
            }
            else if (arg == "--emotion" && i + 1 < argc) {
                std::string emotion = argv[++i];
                if (!conditions.empty()) conditions += " AND ";
                conditions += "emotion='" + emotion + "'";
            }
            else if (arg == "--track" && i + 1 < argc) {
                int track_id = std::stoi(argv[++i]);
                if (!conditions.empty()) conditions += " AND ";
                conditions += "track_id=" + std::to_string(track_id);
            }
            else if (arg == "--export" && i + 1 < argc) {
                export_file = argv[++i];
            }
        }

        if (stats_only) {
            tool.show_statistics();
            return 0;
        }

        auto results = tool.query(conditions, limit);
        tool.print_results(results);

        if (!export_file.empty()) {
            tool.export_csv(results, export_file);
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}