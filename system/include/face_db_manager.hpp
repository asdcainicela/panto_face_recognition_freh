// ============= include/face_db_manager.hpp =============
/*
 * Face Database Manager - Enterprise C++ Solution
 * 
 * ARQUITECTURA:
 * ┌─────────────────┐
 * │  Main Thread    │ → push_face() / request_match()
 * └────────┬────────┘
 *          │ Lock-free queues
 *    ┌─────┴─────┐
 *    │           │
 * ┌──▼───┐   ┌──▼────┐
 * │Writer│   │Matcher│
 * │Pool  │   │Pool   │
 * └──┬───┘   └──┬────┘
 *    │          │
 *    └────┬─────┘
 *         ▼
 *   ┌──────────────┐
 *   │  SQLite WAL  │
 *   │  HNSW Index  │
 *   └──────────────┘
 * 
 * PERFORMANCE GARANTIZADO:
 * - push_face(): <100μs (non-blocking)
 * - request_match(): <200μs + callback
 * - Match search: ~1ms para 100k personas
 * - Batch write: 1000 rostros/segundo
 */

#pragma once
#include "thread_pool.hpp"
#include "vector_index.hpp"
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

// ==================== STRUCTS ====================

struct FaceData {
    int track_id;
    std::vector<float> embedding;      // 512D
    float quality_score;               // [0-1]
    
    // Geometry
    cv::Rect bbox;
    cv::Point2f landmarks[5];
    float yaw_angle;
    float pitch_angle;
    
    // Demographics
    int age;
    std::string gender;
    float gender_confidence;
    
    // Emotion
    std::string emotion;
    float emotion_confidence;
    
    // Metadata
    int64_t timestamp;
};

struct MatchResult {
    std::string person_id;
    std::string name;
    float similarity;
    bool is_new_person;
    int64_t match_time_us;             // Tiempo de búsqueda
};

using MatchCallback = std::function<void(const MatchResult&)>;

// ==================== MAIN CLASS ====================

class FaceDatabaseManager {
public:
    struct Config {
        std::string db_path = "database/faces_v3.db";
        std::string index_path = "database/faces.hnsw";
        float match_threshold = 0.60f;
        float quality_threshold = 0.50f;
        int writer_threads = 2;
        int matcher_threads = 4;
        int batch_size = 50;
        int batch_timeout_ms = 3000;
        bool auto_save_index = true;
        int auto_save_interval_sec = 300;  // 5 min
    };
    
    explicit FaceDatabaseManager(const Config& config = Config{});
    ~FaceDatabaseManager();
    
    // ===== CORE API (THREAD-SAFE, NON-BLOCKING) =====
    
    // Push rostro para procesamiento (async)
    void push_face(const FaceData& face);
    
    // Solicitar match (callback se ejecuta en matcher thread)
    void request_match(int track_id, 
                      const std::vector<float>& embedding,
                      MatchCallback callback);
    
    // Actualizar nombre de persona
    bool update_person_name(const std::string& person_id, const std::string& name);
    
    // Query persona por ID (sync)
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
    void flush();  // Force write pending data
    
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
    std::mutex db_mutex;  // SQLite no es thread-safe sin WAL
    
    // ===== QUEUES =====
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
    
    // ===== CACHE =====
    struct CachedMatch {
        MatchResult result;
        int64_t timestamp;
    };
    std::unordered_map<int, CachedMatch> match_cache;
    std::mutex cache_mutex;
    static constexpr int CACHE_TTL_MS = 30000;  // 30 segundos
    
    // ===== STATS =====
    mutable std::atomic<size_t> stat_cache_hits{0};
    mutable std::atomic<size_t> stat_cache_misses{0};
    mutable std::atomic<size_t> stat_total_matches{0};
    mutable std::atomic<int64_t> stat_total_match_time_us{0};
    
    // ===== WORKER THREADS =====
    std::thread writer_supervisor;
    std::thread matcher_supervisor;
    std::thread maintenance_thread;
    
    // ===== INTERNAL METHODS =====
    
    // Database
    bool init_database();
    bool create_tables();
    
    // Writers
    void writer_supervisor_loop();
    void process_write_batch(std::vector<WriteTask>& batch);
    std::string insert_or_update_person(const FaceData& face, bool is_new);
    std::string generate_person_id();
    
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
    float estimate_quality(const FaceData& face);
    int64_t now_us();
};