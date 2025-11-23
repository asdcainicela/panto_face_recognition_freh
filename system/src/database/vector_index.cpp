// ============= src/database/vector_index.cpp =============
#include "database/vector_index.hpp"  // ⭐ CAMBIO
#include <spdlog/spdlog.h>
#include <fstream>
#include <random>
#include <queue>
#include <set>

// ... resto del código

VectorIndex::VectorIndex(int dim, int M, int ef_construction)
    : dim(dim), M(M), M0(M * 2), ef_construction(ef_construction), max_layer(0)
{
    spdlog::info("🔍 Inicializando HNSW Vector Index");
    spdlog::info("   Dimensión: {}", dim);
    spdlog::info("   M: {}", M);
    spdlog::info("   ef_construction: {}", ef_construction);
}

VectorIndex::~VectorIndex() {
    clear();
}

// ==================== INSERT ====================

void VectorIndex::insert(const std::string& id, const std::vector<float>& vector) {
    std::unique_lock<std::shared_mutex> lock(mutex);
    
    if (entries.find(id) != entries.end()) {
        spdlog::warn("Vector {} already exists, skipping", id);
        return;
    }
    
    VectorEntry entry;
    entry.id = id;
    entry.vector = vector;
    entry.layer = get_random_layer();
    entry.timestamp = std::chrono::system_clock::now().time_since_epoch().count();
    
    // First insertion
    if (entries.empty()) {
        entry_point = id;
        max_layer = entry.layer;
        entries[id] = entry;
        return;
    }
    
    // Search for nearest neighbors at each layer
    std::string current = entry_point;
    
    for (int lc = max_layer; lc > entry.layer; --lc) {
        auto candidates = search_layer(vector, current, lc, 1);
        if (!candidates.empty()) {
            current = candidates[0];
        }
    }
    
    // Insert at layers [0, entry.layer]
    for (int lc = std::min(entry.layer, max_layer); lc >= 0; --lc) {
        auto candidates = search_layer(vector, current, lc, ef_construction);
        
        int M_cur = (lc == 0) ? M0 : M;
        auto neighbors = select_neighbors(vector, candidates, M_cur);
        
        entry.neighbors[lc] = neighbors;
        
        // Bidirectional connections
        for (const auto& neighbor_id : neighbors) {
            if (entries[neighbor_id].neighbors[lc].size() < M_cur) {
                entries[neighbor_id].neighbors[lc].push_back(id);
            } else {
                // Prune connections
                auto& nb_neighbors = entries[neighbor_id].neighbors[lc];
                nb_neighbors.push_back(id);
                nb_neighbors = select_neighbors(entries[neighbor_id].vector, nb_neighbors, M_cur);
            }
        }
    }
    
    entries[id] = entry;
    
    // Update entry point if necessary
    if (entry.layer > max_layer) {
        max_layer = entry.layer;
        entry_point = id;
    }
}

// ==================== SEARCH ====================

std::vector<SearchResult> VectorIndex::search(const std::vector<float>& query, 
                                               int k, 
                                               float threshold) {
    std::shared_lock<std::shared_mutex> lock(mutex);
    
    if (entries.empty()) {
        return {};
    }
    
    // Search from top layer to layer 0
    std::string current = entry_point;
    
    for (int lc = max_layer; lc > 0; --lc) {
        auto candidates = search_layer(query, current, lc, 1);
        if (!candidates.empty()) {
            current = candidates[0];
        }
    }
    
    // Search at layer 0 with larger ef
    auto candidates = search_layer(query, current, 0, std::max(ef_construction, k));
    
    // Convert to SearchResult
    std::vector<SearchResult> results;
    results.reserve(std::min(k, static_cast<int>(candidates.size())));
    
    for (const auto& id : candidates) {
        float dist = distance(query, entries.at(id).vector);
        float sim = 1.0f - dist;
        
        if (sim >= threshold) {
            results.push_back({id, dist, sim});
        }
        
        if (results.size() >= k) break;
    }
    
    return results;
}

// ==================== SEARCH LAYER ====================

std::vector<std::string> VectorIndex::search_layer(
    const std::vector<float>& query,
    const std::string& entry_id,
    int layer,
    int ef)
{
    std::set<std::string> visited;
    
    using Candidate = std::pair<float, std::string>;  // (distance, id)
    auto cmp = [](const Candidate& a, const Candidate& b) { return a.first > b.first; };
    
    std::priority_queue<Candidate, std::vector<Candidate>, decltype(cmp)> candidates(cmp);
    std::priority_queue<Candidate> w;  // Max heap
    
    float d = distance(query, entries.at(entry_id).vector);
    candidates.push({d, entry_id});
    w.push({d, entry_id});
    visited.insert(entry_id);
    
    while (!candidates.empty()) {
        auto [current_dist, current_id] = candidates.top();
        candidates.pop();
        
        if (current_dist > w.top().first) {
            break;
        }
        
        for (const auto& neighbor_id : entries.at(current_id).neighbors[layer]) {
            if (visited.find(neighbor_id) == visited.end()) {
                visited.insert(neighbor_id);
                
                float d_neighbor = distance(query, entries.at(neighbor_id).vector);
                
                if (d_neighbor < w.top().first || w.size() < ef) {
                    candidates.push({d_neighbor, neighbor_id});
                    w.push({d_neighbor, neighbor_id});
                    
                    if (w.size() > ef) {
                        w.pop();
                    }
                }
            }
        }
    }
    
    // Extract results
    std::vector<std::string> results;
    while (!w.empty()) {
        results.push_back(w.top().second);
        w.pop();
    }
    
    std::reverse(results.begin(), results.end());
    return results;
}

// ==================== SELECT NEIGHBORS ====================

std::vector<std::string> VectorIndex::select_neighbors(
    const std::vector<float>& base,
    const std::vector<std::string>& candidates,
    int M)
{
    if (candidates.size() <= M) {
        return candidates;
    }
    
    // Simple heuristic: keep M closest
    std::vector<std::pair<float, std::string>> scored;
    for (const auto& id : candidates) {
        float d = distance(base, entries.at(id).vector);
        scored.push_back({d, id});
    }
    
    std::sort(scored.begin(), scored.end());
    
    std::vector<std::string> selected;
    for (int i = 0; i < M && i < scored.size(); ++i) {
        selected.push_back(scored[i].second);
    }
    
    return selected;
}

// ==================== BATCH INSERT ====================

void VectorIndex::batch_insert(const std::vector<std::pair<std::string, std::vector<float>>>& vectors) {
    spdlog::info("📥 Batch insert: {} vectors", vectors.size());
    
    for (const auto& [id, vec] : vectors) {
        insert(id, vec);
    }
}

// ==================== REMOVE ====================

bool VectorIndex::remove(const std::string& id) {
    std::unique_lock<std::shared_mutex> lock(mutex);
    
    auto it = entries.find(id);
    if (it == entries.end()) {
        return false;
    }
    
    // Remove bidirectional connections
    for (int lc = 0; lc <= it->second.layer; ++lc) {
        for (const auto& neighbor_id : it->second.neighbors[lc]) {
            auto& nb_neighbors = entries[neighbor_id].neighbors[lc];
            nb_neighbors.erase(
                std::remove(nb_neighbors.begin(), nb_neighbors.end(), id),
                nb_neighbors.end()
            );
        }
    }
    
    entries.erase(it);
    
    // Update entry point if necessary
    if (id == entry_point && !entries.empty()) {
        entry_point = entries.begin()->first;
        max_layer = entries.begin()->second.layer;
        
        for (const auto& [eid, entry] : entries) {
            if (entry.layer > max_layer) {
                max_layer = entry.layer;
                entry_point = eid;
            }
        }
    }
    
    return true;
}

// ==================== UPDATE ====================

bool VectorIndex::update(const std::string& id, const std::vector<float>& vector, float quality) {
    std::unique_lock<std::shared_mutex> lock(mutex);
    
    auto it = entries.find(id);
    if (it == entries.end()) {
        return false;
    }
    
    // Only update if better quality
    if (quality > it->second.quality) {
        remove(id);
        lock.unlock();
        insert(id, vector);
        return true;
    }
    
    return false;
}

// ==================== STATS ====================

size_t VectorIndex::size() const {
    std::shared_lock<std::shared_mutex> lock(mutex);
    return entries.size();
}

size_t VectorIndex::memory_usage() const {
    std::shared_lock<std::shared_mutex> lock(mutex);
    
    size_t total = 0;
    for (const auto& [id, entry] : entries) {
        total += entry.vector.size() * sizeof(float);
        total += id.size();
        for (int i = 0; i <= entry.layer; ++i) {
            total += entry.neighbors[i].size() * sizeof(std::string);
        }
    }
    
    return total;
}

void VectorIndex::print_stats() const {
    spdlog::info("=== HNSW Index Stats ===");
    spdlog::info("  Entries: {}", size());
    spdlog::info("  Memory: {:.2f} MB", memory_usage() / 1024.0 / 1024.0);
    spdlog::info("  Max layer: {}", max_layer);
}

void VectorIndex::clear() {
    std::unique_lock<std::shared_mutex> lock(mutex);
    entries.clear();
    entry_point.clear();
    max_layer = 0;
}

// ==================== PERSISTENCE ====================

bool VectorIndex::save(const std::string& filepath) {
    std::shared_lock<std::shared_mutex> lock(mutex);
    
    std::ofstream ofs(filepath, std::ios::binary);
    if (!ofs) return false;
    
    // Write header
    size_t count = entries.size();
    ofs.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    ofs.write(reinterpret_cast<const char*>(&count), sizeof(count));
    
    // Write entries
    for (const auto& [id, entry] : entries) {
        size_t id_len = id.size();
        ofs.write(reinterpret_cast<const char*>(&id_len), sizeof(id_len));
        ofs.write(id.data(), id_len);
        
        ofs.write(reinterpret_cast<const char*>(entry.vector.data()), 
                 entry.vector.size() * sizeof(float));
    }
    
    spdlog::info("✓ Saved index to {}", filepath);
    return true;
}

bool VectorIndex::load(const std::string& filepath) {
    std::unique_lock<std::shared_mutex> lock(mutex);
    
    std::ifstream ifs(filepath, std::ios::binary);
    if (!ifs) return false;
    
    clear();
    
    int loaded_dim;
    size_t count;
    ifs.read(reinterpret_cast<char*>(&loaded_dim), sizeof(loaded_dim));
    ifs.read(reinterpret_cast<char*>(&count), sizeof(count));
    
    if (loaded_dim != dim) {
        spdlog::error("Dimension mismatch: {} vs {}", loaded_dim, dim);
        return false;
    }
    
    lock.unlock();
    
    for (size_t i = 0; i < count; ++i) {
        size_t id_len;
        ifs.read(reinterpret_cast<char*>(&id_len), sizeof(id_len));
        
        std::string id(id_len, '\0');
        ifs.read(&id[0], id_len);
        
        std::vector<float> vec(dim);
        ifs.read(reinterpret_cast<char*>(vec.data()), dim * sizeof(float));
        
        insert(id, vec);
    }
    
    spdlog::info("✓ Loaded {} vectors from {}", count, filepath);
    return true;
}