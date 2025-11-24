// ============= src/database/face_db_manager.cpp - NON-BLOCKING FIX =============
#include "database/face_db_manager.hpp"
#include <spdlog/spdlog.h>
#include <random>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <fstream>

// ... (mantén todo el código anterior hasta push_face)

void FaceDatabaseManager::push_face(const FaceData& face) {
    if (face.quality_score < config.quality_threshold) return;
    
    // ✅ CRÍTICO: NO BLOQUEAR - Si la queue está llena, DESCARTAR
    {
        std::unique_lock<std::mutex> lock(write_mutex, std::try_to_lock);
        
        if (!lock.owns_lock()) {
            // Mutex ocupado, skip silenciosamente
            spdlog::debug("⚠️ Write queue busy, skipping face (quality={:.2f})", 
                         face.quality_score);
            return;
        }
        
        // ✅ Si la queue está llena, DESCARTAR el más antiguo
        if (write_queue.size() >= config.batch_size * 2) {
            spdlog::warn("📦 Write queue FULL ({}), dropping oldest", write_queue.size());
            write_queue.pop();  // Eliminar el más viejo
        }
        
        write_queue.push({face, now_us()});
    }
    
    // NO hay sync ni wait - return inmediatamente
}

void FaceDatabaseManager::request_match(int track_id, const std::vector<float>& embedding, MatchCallback callback) {
    MatchResult cached;
    if (check_cache(track_id, cached)) {
        stat_cache_hits++;
        callback(cached);
        return;
    }
    
    stat_cache_misses++;
    
    // ✅ CRÍTICO: NO BLOQUEAR - Si la queue está llena, usar cache
    {
        std::unique_lock<std::mutex> lock(match_mutex, std::try_to_lock);
        
        if (!lock.owns_lock()) {
            // Mutex ocupado, retornar "Unknown" sin bloquear
            MatchResult result;
            result.person_id = "";
            result.similarity = 0.0f;
            result.is_new_person = true;
            result.name = "Unknown";
            result.match_time_us = 0;
            callback(result);
            return;
        }
        
        // ✅ Si la queue está llena, DESCARTAR request
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

// ==================== WRITER SUPERVISOR (OPTIMIZADO) ====================

void FaceDatabaseManager::writer_supervisor_loop() {
    spdlog::info("📝 Writer supervisor started");
    
    std::vector<WriteTask> batch;
    batch.reserve(config.batch_size);
    auto last_flush = std::chrono::steady_clock::now();
    
    // ✅ TIMEOUT más agresivo para evitar acumulación
    const int FAST_TIMEOUT_MS = 500;  // 500ms en vez de 3000ms
    
    while (running.load()) {
        {
            std::unique_lock<std::mutex> lock(write_mutex, std::defer_lock);
            
            // ✅ Try lock sin bloquear
            if (!lock.try_lock()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }
            
            // Recoger batch rápidamente
            while (!write_queue.empty() && batch.size() < config.batch_size) {
                batch.push_back(write_queue.front());
                write_queue.pop();
            }
        }
        
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_flush).count();
        
        // ✅ Flush más frecuente para evitar acumulación
        if (!batch.empty() && (batch.size() >= config.batch_size / 2 || elapsed >= FAST_TIMEOUT_MS)) {
            // ✅ Log solo si hay muchos pendientes
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
        
        // ✅ Sleep más corto para respuesta rápida
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    
    // Flush final
    if (!batch.empty()) {
        spdlog::info("📝 Final flush: {} faces", batch.size());
        process_write_batch(batch);
    }
    
    spdlog::info("📝 Writer supervisor stopped");
}

// ==================== MATCHER SUPERVISOR (OPTIMIZADO) ====================

void FaceDatabaseManager::matcher_supervisor_loop() {
    spdlog::info("🔍 Matcher supervisor started");
    
    while (running.load()) {
        std::vector<MatchTask> tasks;
        
        {
            std::unique_lock<std::mutex> lock(match_mutex, std::defer_lock);
            
            // ✅ Try lock sin bloquear
            if (!lock.try_lock()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                continue;
            }
            
            // Recoger hasta 20 tasks
            while (!match_queue.empty() && tasks.size() < 20) {
                tasks.push_back(match_queue.front());
                match_queue.pop();
            }
        }
        
        // ✅ Procesar tasks en paralelo
        for (auto& task : tasks) {
            matcher_pool->submit([this, task]() {
                process_match_task(task);
            });
        }
        
        // ✅ Sleep muy corto para baja latencia
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    
    spdlog::info("🔍 Matcher supervisor stopped");
}

// ==================== PROCESS WRITE BATCH (OPTIMIZADO) ====================

void FaceDatabaseManager::process_write_batch(std::vector<WriteTask>& batch) {
    if (batch.empty()) return;
    
    auto batch_start = std::chrono::high_resolution_clock::now();
    
    std::lock_guard<std::mutex> lock(db_mutex);
    
    // ✅ BEGIN IMMEDIATE para evitar lock wait
    sqlite3_exec(db, "BEGIN IMMEDIATE;", nullptr, nullptr, nullptr);
    
    int inserted = 0;
    int updated = 0;
    
    for (auto& task : batch) {
        auto& face = task.face;
        
        // ✅ Search en index sin lock prolongado
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
    
    // ✅ Log solo si el batch fue lento
    if (batch_ms > 100) {
        spdlog::warn("⏱️ Slow batch write: {:.1f}ms ({} faces, {} new, {} updated)", 
                    batch_ms, batch.size(), inserted, updated);
    }
}

// ... (resto del código sin cambios)

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

int64_t FaceDatabaseManager::now_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();
}