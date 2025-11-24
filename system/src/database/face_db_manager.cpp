// ============= src/database/face_db_manager.cpp (COMPLETO) =============
#include "database/face_db_manager.hpp"
#include <spdlog/spdlog.h>
#include <random>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <fstream>

FaceDatabaseManager::FaceDatabaseManager(const Config& config)
    : config(config), db(nullptr)
{
    spdlog::info("🏢 Inicializando Face Database Manager (Enterprise)");
    spdlog::info("   DB: {}", config.db_path);
    spdlog::info("   Match threshold: {:.2f}", config.match_threshold);
    spdlog::info("   Writer threads: {}", config.writer_threads);
    spdlog::info("   Matcher threads: {}", config.matcher_threads);
    
    if (!init_database()) {
        throw std::runtime_error("Failed to init database");
    }
    
    index = std::make_unique<VectorIndex>(512, 16, 200);
    
    try {
        if (std::ifstream(config.index_path).good()) {
            spdlog::info("📂 Loading existing HNSW index...");
            if (index->load(config.index_path)) {
                spdlog::info("✓ Index loaded: {} vectors", index->size());
            }
        } else {
            spdlog::info("📥 Building index from database...");
            
            const char* sql = "SELECT person_id, embedding FROM face_embeddings ORDER BY quality_score DESC";
            sqlite3_stmt* stmt;
            
            if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) == SQLITE_OK) {
                std::vector<std::pair<std::string, std::vector<float>>> vectors;
                std::unordered_set<std::string> added;
                
                while (sqlite3_step(stmt) == SQLITE_ROW) {
                    std::string person_id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
                    
                    if (added.find(person_id) != added.end()) continue;
                    added.insert(person_id);
                    
                    const unsigned char* blob = static_cast<const unsigned char*>(sqlite3_column_blob(stmt, 1));
                    int blob_size = sqlite3_column_bytes(stmt, 1);
                    
                    std::vector<float> emb(blob_size / sizeof(float));
                    std::memcpy(emb.data(), blob, blob_size);
                    
                    vectors.emplace_back(person_id, emb);
                }
                
                sqlite3_finalize(stmt);
                
                if (!vectors.empty()) {
                    spdlog::info("   Building index for {} persons...", vectors.size());
                    index->batch_insert(vectors);
                    spdlog::info("✓ Index built");
                } else {
                    spdlog::info("   Empty database, starting fresh");
                }
            }
        }
    } catch (const std::exception& e) {
        spdlog::error("Index load error: {}", e.what());
        spdlog::info("   Starting with empty index");
    }
    
    writer_pool = std::make_unique<ThreadPool>(config.writer_threads);
    matcher_pool = std::make_unique<ThreadPool>(config.matcher_threads);
}

FaceDatabaseManager::~FaceDatabaseManager() {
    stop();
    if (db) sqlite3_close(db);
}

bool FaceDatabaseManager::init_database() {
    int rc = sqlite3_open(config.db_path.c_str(), &db);
    if (rc != SQLITE_OK) {
        spdlog::error("Cannot open database: {}", sqlite3_errmsg(db));
        return false;
    }
    
    char* err = nullptr;
    sqlite3_exec(db, "PRAGMA journal_mode=WAL;", nullptr, nullptr, &err);
    sqlite3_exec(db, "PRAGMA synchronous=NORMAL;", nullptr, nullptr, &err);
    sqlite3_exec(db, "PRAGMA cache_size=10000;", nullptr, nullptr, &err);
    sqlite3_exec(db, "PRAGMA temp_store=MEMORY;", nullptr, nullptr, &err);
    
    return create_tables();
}

bool FaceDatabaseManager::create_tables() {
    const char* sql = R"(
        CREATE TABLE IF NOT EXISTS persons (
            person_id TEXT PRIMARY KEY,
            name TEXT DEFAULT 'Unknown',
            first_seen INTEGER,
            last_seen INTEGER,
            total_faces INTEGER DEFAULT 0,
            best_quality REAL DEFAULT 0.0,
            notes TEXT
        );
        
        CREATE TABLE IF NOT EXISTS face_embeddings (
            embedding_id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id TEXT NOT NULL,
            embedding BLOB NOT NULL,
            quality_score REAL NOT NULL,
            face_width INTEGER,
            face_height INTEGER,
            yaw_angle REAL,
            pitch_angle REAL,
            age INTEGER,
            gender TEXT,
            gender_confidence REAL,
            emotion TEXT,
            emotion_confidence REAL,
            captured_at INTEGER,
            FOREIGN KEY (person_id) REFERENCES persons(person_id)
        );
        
        CREATE INDEX IF NOT EXISTS idx_person_last_seen ON persons(last_seen DESC);
        CREATE INDEX IF NOT EXISTS idx_embeddings_person ON face_embeddings(person_id);
        CREATE INDEX IF NOT EXISTS idx_embeddings_quality ON face_embeddings(quality_score DESC);
    )";
    
    char* err = nullptr;
    int rc = sqlite3_exec(db, sql, nullptr, nullptr, &err);
    
    if (rc != SQLITE_OK) {
        spdlog::error("SQL error: {}", err);
        sqlite3_free(err);
        return false;
    }
    
    return true;
}

void FaceDatabaseManager::start() {
    if (running.load()) {
        spdlog::warn("Already running");
        return;
    }
    
    running = true;
    writer_supervisor = std::thread(&FaceDatabaseManager::writer_supervisor_loop, this);
    matcher_supervisor = std::thread(&FaceDatabaseManager::matcher_supervisor_loop, this);
    maintenance_thread = std::thread(&FaceDatabaseManager::maintenance_loop, this);
    
    spdlog::info("✓ Database manager started");
}

void FaceDatabaseManager::stop() {
    if (!running.load()) return;
    
    spdlog::info("🛑 Stopping database manager...");
    running = false;
    
    flush();
    
    if (writer_supervisor.joinable()) writer_supervisor.join();
    if (matcher_supervisor.joinable()) matcher_supervisor.join();
    if (maintenance_thread.joinable()) maintenance_thread.join();
    
    if (config.auto_save_index) save_index_to_disk();
    
    writer_pool->stop();
    matcher_pool->stop();
    
    spdlog::info("✓ Database manager stopped");
}

void FaceDatabaseManager::flush() {
    while (true) {
        size_t pending_w, pending_m;
        {
            std::lock_guard<std::mutex> lw(write_mutex);
            std::lock_guard<std::mutex> lm(match_mutex);
            pending_w = write_queue.size();
            pending_m = match_queue.size();
        }
        
        if (pending_w == 0 && pending_m == 0) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    writer_pool->wait_all();
    matcher_pool->wait_all();
}

// ✅ PUSH FACE - NO BLOQUEANTE
void FaceDatabaseManager::push_face(const FaceData& face) {
    if (face.quality_score < config.quality_threshold) return;
    
    {
        std::unique_lock<std::mutex> lock(write_mutex, std::try_to_lock);
        
        if (!lock.owns_lock()) {
            spdlog::debug("⚠️ Write queue busy, skipping face");
            return;
        }
        
        if (write_queue.size() >= config.batch_size * 2) {
            spdlog::warn("📦 Write queue FULL ({}), dropping oldest", write_queue.size());
            write_queue.pop();
        }
        
        write_queue.push({face, now_us()});
    }
}

// ✅ REQUEST MATCH - NO BLOQUEANTE
void FaceDatabaseManager::request_match(int track_id, const std::vector<float>& embedding, MatchCallback callback) {
    MatchResult cached;
    if (check_cache(track_id, cached)) {
        stat_cache_hits++;
        callback(cached);
        return;
    }
    
    stat_cache_misses++;
    
    {
        std::unique_lock<std::mutex> lock(match_mutex, std::try_to_lock);
        
        if (!lock.owns_lock()) {
            MatchResult result;
            result.person_id = "";
            result.similarity = 0.0f;
            result.is_new_person = true;
            result.name = "Unknown";
            result.match_time_us = 0;
            callback(result);
            return;
        }
        
        if (match_queue.size() >= 100) {
            spdlog::warn("🔍 Match queue FULL ({}), using cache fallback", match_queue.size());
            MatchResult result;
            result.person_id = "";
            result.similarity = 0.0f;
            result.is_new_person = true;
            result.name = "Unknown";
            result.match_time_us = 0;
            callback(result);
            return;
        }
        
        match_queue.push({track_id, embedding, callback, now_us()});
    }
}

// ✅ WRITER SUPERVISOR - OPTIMIZADO
void FaceDatabaseManager::writer_supervisor_loop() {
    spdlog::info("📝 Writer supervisor started");
    
    std::vector<WriteTask> batch;
    batch.reserve(config.batch_size);
    auto last_flush = std::chrono::steady_clock::now();
    
    const int FAST_TIMEOUT_MS = 500;
    
    while (running.load()) {
        {
            std::unique_lock<std::mutex> lock(write_mutex, std::defer_lock);
            
            if (!lock.try_lock()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }
            
            while (!write_queue.empty() && batch.size() < config.batch_size) {
                batch.push_back(write_queue.front());
                write_queue.pop();
            }
        }
        
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_flush).count();
        
        if (!batch.empty() && (batch.size() >= config.batch_size / 2 || elapsed >= FAST_TIMEOUT_MS)) {
            if (batch.size() > 10) {
                spdlog::debug("📝 Flushing batch: {} faces", batch.size());
            }
            
            writer_pool->submit([this, batch = std::move(batch)]() mutable {
                process_write_batch(batch);
            });
            
            batch.clear();
            batch.reserve(config.batch_size);
            last_flush = now;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    
    if (!batch.empty()) {
        spdlog::info("📝 Final flush: {} faces", batch.size());
        process_write_batch(batch);
    }
    
    spdlog::info("📝 Writer supervisor stopped");
}

// ✅ MATCHER SUPERVISOR - OPTIMIZADO
void FaceDatabaseManager::matcher_supervisor_loop() {
    spdlog::info("🔍 Matcher supervisor started");
    
    while (running.load()) {
        std::vector<MatchTask> tasks;
        
        {
            std::unique_lock<std::mutex> lock(match_mutex, std::defer_lock);
            
            if (!lock.try_lock()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                continue;
            }
            
            while (!match_queue.empty() && tasks.size() < 20) {
                tasks.push_back(match_queue.front());
                match_queue.pop();
            }
        }
        
        for (auto& task : tasks) {
            matcher_pool->submit([this, task]() {
                process_match_task(task);
            });
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    
    spdlog::info("🔍 Matcher supervisor stopped");
}

void FaceDatabaseManager::process_write_batch(std::vector<WriteTask>& batch) {
    if (batch.empty()) return;
    
    auto batch_start = std::chrono::high_resolution_clock::now();
    
    std::lock_guard<std::mutex> lock(db_mutex);
    
    sqlite3_exec(db, "BEGIN IMMEDIATE;", nullptr, nullptr, nullptr);
    
    int inserted = 0;
    int updated = 0;
    
    for (auto& task : batch) {
        auto& face = task.face;
        auto results = index->search(face.embedding, 1, config.match_threshold);
        
        std::string person_id;
        bool is_new = results.empty();
        
        if (!is_new) {
            person_id = results[0].id;
            auto info = get_person_info(person_id);
            if (face.quality_score > info.best_quality) {
                index->update(person_id, face.embedding, face.quality_score);
            }
            updated++;
        } else {
            person_id = generate_person_id();
            index->insert(person_id, face.embedding);
            inserted++;
        }
        
        insert_or_update_person(face, is_new, person_id);
    }
    
    sqlite3_exec(db, "COMMIT;", nullptr, nullptr, nullptr);
    
    auto batch_end = std::chrono::high_resolution_clock::now();
    auto batch_ms = std::chrono::duration<double, std::milli>(batch_end - batch_start).count();
    
    if (batch_ms > 100) {
        spdlog::warn("⏱️ Slow batch write: {:.1f}ms ({} faces, {} new, {} updated)", 
                    batch_ms, batch.size(), inserted, updated);
    }
}

void FaceDatabaseManager::process_match_task(const MatchTask& task) {
    auto start = std::chrono::high_resolution_clock::now();
    auto results = index->search(task.embedding, 1, config.match_threshold);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    MatchResult result;
    result.match_time_us = elapsed_us;
    
    if (!results.empty()) {
        result.person_id = results[0].id;
        result.similarity = results[0].similarity;
        result.is_new_person = false;
        auto info = get_person_info(result.person_id);
        result.name = info.name;
    } else {
        result.person_id = "";
        result.similarity = 0.0f;
        result.is_new_person = true;
        result.name = "Unknown";
    }
    
    stat_total_matches++;
    stat_total_match_time_us += elapsed_us;
    
    update_cache(task.track_id, result);
    
    try {
        task.callback(result);
    } catch (const std::exception& e) {
        spdlog::error("Match callback exception: {}", e.what());
    }
}

void FaceDatabaseManager::insert_or_update_person(const FaceData& face, bool is_new, const std::string& person_id) {
    if (is_new) {
        const char* sql = "INSERT INTO persons (person_id, first_seen, last_seen, total_faces, best_quality) VALUES (?, ?, ?, 1, ?)";
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) == SQLITE_OK) {
            sqlite3_bind_text(stmt, 1, person_id.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_bind_int64(stmt, 2, face.timestamp);
            sqlite3_bind_int64(stmt, 3, face.timestamp);
            sqlite3_bind_double(stmt, 4, face.quality_score);
            sqlite3_step(stmt);
            sqlite3_finalize(stmt);
        }
    } else {
        const char* sql = "UPDATE persons SET last_seen=?, total_faces=total_faces+1, best_quality=MAX(best_quality,?) WHERE person_id=?";
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) == SQLITE_OK) {
            sqlite3_bind_int64(stmt, 1, face.timestamp);
            sqlite3_bind_double(stmt, 2, face.quality_score);
            sqlite3_bind_text(stmt, 3, person_id.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_step(stmt);
            sqlite3_finalize(stmt);
        }
    }
    
    const char* sql2 = R"(
        INSERT INTO face_embeddings (person_id, embedding, quality_score, face_width, face_height,
                                     yaw_angle, pitch_angle, age, gender, gender_confidence, emotion, emotion_confidence, captured_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    )";
    sqlite3_stmt* stmt2;
    if (sqlite3_prepare_v2(db, sql2, -1, &stmt2, nullptr) == SQLITE_OK) {
        std::vector<unsigned char> blob(face.embedding.size() * sizeof(float));
        std::memcpy(blob.data(), face.embedding.data(), blob.size());
        
        sqlite3_bind_text(stmt2, 1, person_id.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_blob(stmt2, 2, blob.data(), blob.size(), SQLITE_TRANSIENT);
        sqlite3_bind_double(stmt2, 3, face.quality_score);
        sqlite3_bind_int(stmt2, 4, face.bbox.width);
        sqlite3_bind_int(stmt2, 5, face.bbox.height);
        sqlite3_bind_double(stmt2, 6, face.yaw_angle);
        sqlite3_bind_double(stmt2, 7, face.pitch_angle);
        sqlite3_bind_int(stmt2, 8, face.age);
        sqlite3_bind_text(stmt2, 9, face.gender.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_double(stmt2, 10, face.gender_confidence);
        sqlite3_bind_text(stmt2, 11, face.emotion.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_double(stmt2, 12, face.emotion_confidence);
        sqlite3_bind_int64(stmt2, 13, face.timestamp);
        sqlite3_step(stmt2);
        sqlite3_finalize(stmt2);
    }
}

std::string FaceDatabaseManager::generate_person_id() {
    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 0xFFFF);
    
    std::ostringstream oss;
    oss << "p_" << std::hex << (ms & 0xFFFFFF) << dis(gen);
    return oss.str();
}

void FaceDatabaseManager::maintenance_loop() {
    spdlog::info("🔧 Maintenance thread started");
    
    auto last_save = std::chrono::steady_clock::now();
    
    while (running.load()) {
        std::this_thread::sleep_for(std::chrono::seconds(60));
        
        clear_expired_cache();
        
        if (config.auto_save_index) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_save).count();
            
            if (elapsed >= config.auto_save_interval_sec) {
                save_index_to_disk();
                last_save = now;
            }
        }
    }
    
    spdlog::info("🔧 Maintenance thread stopped");
}

void FaceDatabaseManager::save_index_to_disk() {
    spdlog::info("💾 Saving HNSW index...");
    if (index->save(config.index_path)) {
        spdlog::info("✓ Index saved: {} vectors", index->size());
    }
}

void FaceDatabaseManager::clear_expired_cache() {
    std::lock_guard<std::mutex> lock(cache_mutex);
    
    int64_t now = now_us();
    auto it = match_cache.begin();
    
    while (it != match_cache.end()) {
        if ((now - it->second.timestamp) > CACHE_TTL_MS * 1000) {
            it = match_cache.erase(it);
        } else {
            ++it;
        }
    }
}

bool FaceDatabaseManager::check_cache(int track_id, MatchResult& result) {
    std::lock_guard<std::mutex> lock(cache_mutex);
    
    auto it = match_cache.find(track_id);
    if (it == match_cache.end()) return false;
    
    int64_t now = now_us();
    if ((now - it->second.timestamp) < CACHE_TTL_MS * 1000) {
        result = it->second.result;
        return true;
    }
    
    return false;
}

void FaceDatabaseManager::update_cache(int track_id, const MatchResult& result) {
    std::lock_guard<std::mutex> lock(cache_mutex);
    match_cache[track_id] = {result, now_us()};
}

FaceDatabaseManager::PersonInfo FaceDatabaseManager::get_person_info(const std::string& person_id) {
    std::lock_guard<std::mutex> lock(db_mutex);
    
    PersonInfo info;
    info.person_id = person_id;
    
    const char* sql = "SELECT name, total_faces, best_quality, first_seen, last_seen FROM persons WHERE person_id=?";
    sqlite3_stmt* stmt;
    
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) == SQLITE_OK) {
        sqlite3_bind_text(stmt, 1, person_id.c_str(), -1, SQLITE_TRANSIENT);
        
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            info.name = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
            info.total_faces = sqlite3_column_int(stmt, 1);
            info.best_quality = sqlite3_column_double(stmt, 2);
        }
        
        sqlite3_finalize(stmt);
    }
    
    return info;
}

bool FaceDatabaseManager::update_person_name(const std::string& person_id, const std::string& name) {
    std::lock_guard<std::mutex> lock(db_mutex);
    
    const char* sql = "UPDATE persons SET name=? WHERE person_id=?";
    sqlite3_stmt* stmt;
    
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) == SQLITE_OK) {
        sqlite3_bind_text(stmt, 1, name.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 2, person_id.c_str(), -1, SQLITE_TRANSIENT);
        int rc = sqlite3_step(stmt);
        sqlite3_finalize(stmt);
        return rc == SQLITE_DONE;
    }
    
    return false;
}

FaceDatabaseManager::Stats FaceDatabaseManager::get_stats() const {
    Stats stats;
    
    {
        std::lock_guard<std::mutex> lw(write_mutex);
        stats.pending_writes = write_queue.size();
    }
    
    {
        std::lock_guard<std::mutex> lm(match_mutex);
        stats.pending_matches = match_queue.size();
    }
    
    stats.cache_hits = stat_cache_hits.load();
    stats.cache_misses = stat_cache_misses.load();
    
    size_t total = stat_total_matches.load();
    stats.avg_match_time_ms = total > 0 ? (stat_total_match_time_us.load() / (double)total / 1000.0) : 0.0;
    
    stats.total_embeddings = index->size();
    stats.index_memory_mb = index->memory_usage() / 1024.0 / 1024.0;
    
    {
        std::lock_guard<std::mutex> lock(db_mutex);
        const char* sql = "SELECT COUNT(*) FROM persons";
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) == SQLITE_OK) {
            if (sqlite3_step(stmt) == SQLITE_ROW) {
                stats.total_persons = sqlite3_column_int(stmt, 0);
            }
            sqlite3_finalize(stmt);
        }
    }
    
    return stats;
}

void FaceDatabaseManager::print_stats() const {
    auto stats = get_stats();
    spdlog::info("=== Database Manager Stats ===");
    spdlog::info("  Persons: {}", stats.total_persons);
    spdlog::info("  Embeddings: {}", stats.total_embeddings);
    spdlog::info("  Index memory: {:.2f} MB", stats.index_memory_mb);
    spdlog::info("  Cache hits: {} / {}", stats.cache_hits, stats.cache_hits + stats.cache_misses);
    spdlog::info("  Avg match time: {:.2f} ms", stats.avg_match_time_ms);
}

int64_t FaceDatabaseManager::now_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();
}