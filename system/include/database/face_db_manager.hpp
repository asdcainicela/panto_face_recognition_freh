// ============= include/database/face_db_manager_fixed.hpp =============
/*
 * Face Database Manager - FIXED VERSION
 * 
 * CAMBIOS CRÍTICOS:
 * ✅ Separación completa de locks (index vs db vs cache)
 * ✅ Lock-free queues con límites estrictos
 * ✅ Writer no llama a index->search() bajo db_mutex
 * ✅ Timeout automático para operaciones bloqueadas
 * ✅ Backpressure explícito cuando queues están llenas
 */

#pragma once
#include "database/thread_pool.hpp"
#include "database/vector_index.hpp"
#include <string>
#include <vector>
#include <queue>
#include <mutex>
#include <atomic>
#include <functional>
#include <unordered_map>
#include <chrono>
#include <sqlite3.h>
#include <opencv2/opencv.hpp>
#include <condition_variable>

struct FaceData {
    int track_id;
    std::vector<float> embedding;
    float quality_score;
    cv::Rect bbox;
    cv::Point2f landmarks[5];
    float yaw_angle;
    float pitch_angle;
    int age;
    std::string gender;
    float gender_confidence;
    std::string emotion;
    float emotion_confidence;
    int64_t timestamp;
};

struct MatchResult {
    std::string person_id;
    std::string name;
    float similarity;
    bool is_new_person;
    int64_t match_time_us;
};

using MatchCallback = std::function<void(const MatchResult&)>;

// ==================== FIXED DATABASE MANAGER ====================

class FaceDatabaseManager {
public:
    struct Config {
        std::string db_path;
        std::string index_path;
        float match_threshold;
        float quality_threshold;
        int writer_threads;
        int matcher_threads;
        int max_write_queue;      // ✅ Límite estricto
        int max_match_queue;      // ✅ Límite estricto
        int batch_timeout_ms;
        bool auto_save_index;
        int auto_save_interval_sec;
        
        Config() 
            : db_path("database/faces_v3.db"),
              index_path("database/faces.hnsw"),
              match_threshold(0.60f),
              quality_threshold(0.50f),
              writer_threads(2),
              matcher_threads(4),
              max_write_queue(100),   // ✅ Máximo 100 writes pendientes
              max_match_queue(50),    // ✅ Máximo 50 matches pendientes
              batch_timeout_ms(3000),
              auto_save_index(true),
              auto_save_interval_sec(300) {}
    };
    
    explicit FaceDatabaseManager(const Config& config = Config());
    ~FaceDatabaseManager();
    
    // ===== CORE API (THREAD-SAFE, NON-BLOCKING CON BACKPRESSURE) =====
    
    // ✅ Retorna false si queue está llena (backpressure)
    bool push_face(const FaceData& face);
    
    // ✅ Retorna false si queue está llena (callback se ejecuta con error)
    bool request_match(int track_id, 
                      const std::vector<float>& embedding,
                      MatchCallback callback);
    
    bool update_person_name(const std::string& person_id, const std::string& name);
    
    struct PersonInfo {
        std::string person_id;
        std::string name;
        int total_faces;
        float best_quality;
        std::string first_seen;
        std::string last_seen;
    };
    PersonInfo get_person_info(const std::string& person_id);
    
    // ===== STATS & MONITORING =====
    
    struct Stats {
        size_t total_persons;
        size_t total_embeddings;
        size_t pending_writes;
        size_t pending_matches;
        size_t dropped_writes;    // ✅ Contador de writes rechazados
        size_t dropped_matches;   // ✅ Contador de matches rechazados
        size_t cache_hits;
        size_t cache_misses;
        double avg_match_time_ms;
        size_t index_memory_mb;
    };
    Stats get_stats() const;
    
    void print_stats() const;
    
    // ===== LIFECYCLE =====
    
    void start();
    void stop();
    void flush();
    
    bool is_running() const { return running.load(); }
    
private:
    Config config;
    std::atomic<bool> running{false};
    
    // ===== THREAD POOLS =====
    std::unique_ptr<ThreadPool> writer_pool;
    std::unique_ptr<ThreadPool> matcher_pool;
    
    // ===== STORAGE =====
    sqlite3* db;
    std::unique_ptr<VectorIndex> index;
    
    // ✅ LOCKS SEPARADOS (nunca se adquieren juntos)
    mutable std::mutex db_mutex;      // Solo para SQLite
    mutable std::mutex index_mutex;   // Solo para VectorIndex (en vez de shared_mutex interno)
    mutable std::mutex cache_mutex;   // Solo para cache
    
    // ===== QUEUES CON CONDITION VARIABLES =====
    struct WriteTask {
        FaceData face;
        int64_t enqueue_time;
    };
    
    struct MatchTask {
        int track_id;
        std::vector<float> embedding;
        MatchCallback callback;
        int64_t enqueue_time;
    };
    
    std::queue<WriteTask> write_queue;
    std::queue<MatchTask> match_queue;
    std::mutex write_mutex;
    std::mutex match_mutex;
    std::condition_variable write_cv;   // ✅ Para despertar writer
    std::condition_variable match_cv;   // ✅ Para despertar matcher
    
    // ===== CACHE =====
    struct CachedMatch {
        MatchResult result;
        int64_t timestamp;
    };
    std::unordered_map<int, CachedMatch> match_cache;
    static constexpr int CACHE_TTL_MS = 30000;
    
    // ===== STATS =====
    mutable std::atomic<size_t> stat_cache_hits{0};
    mutable std::atomic<size_t> stat_cache_misses{0};
    mutable std::atomic<size_t> stat_total_matches{0};
    mutable std::atomic<int64_t> stat_total_match_time_us{0};
    mutable std::atomic<size_t> stat_dropped_writes{0};   // ✅ Nuevo
    mutable std::atomic<size_t> stat_dropped_matches{0};  // ✅ Nuevo
    
    // ===== WORKER THREADS =====
    std::thread writer_supervisor;
    std::thread matcher_supervisor;
    std::thread maintenance_thread;
    
    // ===== INTERNAL METHODS =====
    
    // Database
    bool init_database();
    bool create_tables();
    
    // Writers (✅ CORREGIDO: No llama index->search bajo db_mutex)
    void writer_supervisor_loop();
    void process_write_batch(std::vector<WriteTask>& batch);
    void insert_or_update_person(const FaceData& face, bool is_new, const std::string& person_id);
    std::string generate_person_id();
    
    // ✅ NUEVO: Pre-match separado (sin db_mutex)
    struct PreMatchResult {
        bool is_new;
        std::string person_id;
        float similarity;
    };
    PreMatchResult pre_match_face(const std::vector<float>& embedding);
    
    // Matchers
    void matcher_supervisor_loop();
    void process_match_task(const MatchTask& task);
    
    // Maintenance
    void maintenance_loop();
    void save_index_to_disk();
    void clear_expired_cache();
    
    // Helpers
    bool check_cache(int track_id, MatchResult& result);
    void update_cache(int track_id, const MatchResult& result);
    int64_t now_us();
};