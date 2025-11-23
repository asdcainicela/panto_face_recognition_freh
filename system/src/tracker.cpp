// ============= src/tracker.cpp =============
#include "tracker.hpp"
#include "detector_optimized.hpp"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <numeric>  // para std::iota


TrackedFace::TrackedFace() 
    : id(-1), confidence(0.0f), age(0), hits(0), 
      time_since_update(0), kf_initialized(false),
      is_recognized(false), name("Unknown"),
      emotion("Unknown"), emotion_confidence(0.0f),
      age_years(0), gender("Unknown"), gender_confidence(0.0f),
      best_quality(0.0f), person_id("")
{
    for (int i = 0; i < 5; i++) {
        landmarks[i] = cv::Point2f(0, 0);
    }
    embedding.clear();
}

void TrackedFace::init_kalman() {
    // State: [x, y, w, h, dx, dy, dw, dh]
    // Measurement: [x, y, w, h]
    kf.init(8, 4, 0);
    
    // Transition matrix (constant velocity model)
    kf.transitionMatrix = (cv::Mat_<float>(8, 8) << 
        1, 0, 0, 0, 1, 0, 0, 0,   // x' = x + dx
        0, 1, 0, 0, 0, 1, 0, 0,   // y' = y + dy
        0, 0, 1, 0, 0, 0, 1, 0,   // w' = w + dw
        0, 0, 0, 1, 0, 0, 0, 1,   // h' = h + dh
        0, 0, 0, 0, 1, 0, 0, 0,   // dx' = dx
        0, 0, 0, 0, 0, 1, 0, 0,   // dy' = dy
        0, 0, 0, 0, 0, 0, 1, 0,   // dw' = dw
        0, 0, 0, 0, 0, 0, 0, 1    // dh' = dh
    );
    
    // Measurement matrix
    kf.measurementMatrix = (cv::Mat_<float>(4, 8) << 
        1, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0
    );
    
    // Process noise covariance
    cv::setIdentity(kf.processNoiseCov, cv::Scalar::all(1e-2));
    
    // Measurement noise covariance
    cv::setIdentity(kf.measurementNoiseCov, cv::Scalar::all(1e-1));
    
    // Error covariance
    cv::setIdentity(kf.errorCovPost, cv::Scalar::all(1));
    
    // Initialize state
    kf.statePost.at<float>(0) = box.x + box.width / 2.0f;   // x
    kf.statePost.at<float>(1) = box.y + box.height / 2.0f;  // y
    kf.statePost.at<float>(2) = box.width;                   // w
    kf.statePost.at<float>(3) = box.height;                  // h
    kf.statePost.at<float>(4) = 0;  // dx
    kf.statePost.at<float>(5) = 0;  // dy
    kf.statePost.at<float>(6) = 0;  // dw
    kf.statePost.at<float>(7) = 0;  // dh
    
    kf_initialized = true;
}

void TrackedFace::predict() {
    if (!kf_initialized) {
        init_kalman();
        return;
    }
    
    cv::Mat prediction = kf.predict();
    
    float cx = prediction.at<float>(0);
    float cy = prediction.at<float>(1);
    float w = prediction.at<float>(2);
    float h = prediction.at<float>(3);
    
    box.x = static_cast<int>(cx - w / 2.0f);
    box.y = static_cast<int>(cy - h / 2.0f);
    box.width = static_cast<int>(w);
    box.height = static_cast<int>(h);
}

void TrackedFace::update(const cv::Rect& measurement) {
    if (!kf_initialized) {
        box = measurement;
        init_kalman();
        return;
    }
    
    cv::Mat m = (cv::Mat_<float>(4, 1) << 
        measurement.x + measurement.width / 2.0f,
        measurement.y + measurement.height / 2.0f,
        measurement.width,
        measurement.height
    );
    
    kf.correct(m);
    
    box = measurement;
}

// ==================== FaceTracker ====================

FaceTracker::FaceTracker(float iou_threshold, int max_age, int min_hits)
    : next_id(1), iou_threshold(iou_threshold), 
      max_age(max_age), min_hits(min_hits) 
{
    spdlog::info("🎯 Face Tracker initialized");
    spdlog::info("   IoU threshold: {}", iou_threshold);
    spdlog::info("   Max age: {} frames", max_age);
    spdlog::info("   Min hits: {}", min_hits);
}

float FaceTracker::calculate_iou(const cv::Rect& a, const cv::Rect& b) {
    float intersection = (a & b).area();
    float union_area = a.area() + b.area() - intersection;
    
    if (union_area <= 0) return 0.0f;
    return intersection / union_area;
}

void FaceTracker::linear_assignment(
    const std::vector<std::vector<float>>& iou_matrix,
    const std::vector<int>& detection_indices,
    const std::vector<int>& track_indices,
    std::vector<std::pair<int, int>>& matches,
    std::vector<int>& unmatched_dets,
    std::vector<int>& unmatched_tracks)
{
    matches.clear();
    unmatched_dets.clear();
    unmatched_tracks.clear();
    
    if (detection_indices.empty() || track_indices.empty()) {
        unmatched_dets = detection_indices;
        unmatched_tracks = track_indices;
        return;
    }
    
    // Greedy assignment basado en IoU máximo
    std::vector<bool> det_matched(detection_indices.size(), false);
    std::vector<bool> track_matched(track_indices.size(), false);
    
    // Buscar mejores matches
    for (size_t t = 0; t < track_indices.size(); t++) {
        float max_iou = iou_threshold;
        int best_det = -1;
        
        for (size_t d = 0; d < detection_indices.size(); d++) {
            if (det_matched[d]) continue;
            
            float iou = iou_matrix[d][t];
            if (iou > max_iou) {
                max_iou = iou;
                best_det = d;
            }
        }
        
        if (best_det >= 0) {
            matches.push_back({detection_indices[best_det], track_indices[t]});
            det_matched[best_det] = true;
            track_matched[t] = true;
        }
    }
    
    // Recolectar unmatched
    for (size_t d = 0; d < detection_indices.size(); d++) {
        if (!det_matched[d]) {
            unmatched_dets.push_back(detection_indices[d]);
        }
    }
    
    for (size_t t = 0; t < track_indices.size(); t++) {
        if (!track_matched[t]) {
            unmatched_tracks.push_back(track_indices[t]);
        }
    }
}

void FaceTracker::associate_detections_to_tracks(
    const std::vector<Detection>& detections,
    const std::vector<int>& track_indices,
    std::vector<std::pair<int, int>>& matches,
    std::vector<int>& unmatched_dets,
    std::vector<int>& unmatched_tracks)
{
    if (track_indices.empty()) {
        unmatched_dets.clear();
        for (size_t i = 0; i < detections.size(); i++) {
            unmatched_dets.push_back(i);
        }
        return;
    }
    
    // Calcular IoU matrix
    std::vector<std::vector<float>> iou_matrix(
        detections.size(), 
        std::vector<float>(track_indices.size(), 0.0f)
    );
    
    for (size_t d = 0; d < detections.size(); d++) {
        for (size_t t = 0; t < track_indices.size(); t++) {
            iou_matrix[d][t] = calculate_iou(
                detections[d].box, 
                tracks[track_indices[t]].box
            );
        }
    }
    
    // Assignment
    std::vector<int> det_indices(detections.size());
    std::iota(det_indices.begin(), det_indices.end(), 0);
    
    linear_assignment(iou_matrix, det_indices, track_indices, 
                     matches, unmatched_dets, unmatched_tracks);
}

std::vector<TrackedFace> FaceTracker::update(const std::vector<Detection>& detections) {
    // 1. Separar detecciones por confianza (ByteTrack strategy)
    std::vector<Detection> high_conf, low_conf;
    for (const auto& det : detections) {
        if (det.confidence >= 0.5f) {
            high_conf.push_back(det);
        } else if (det.confidence >= 0.3f) {
            low_conf.push_back(det);
        }
    }
    
    // 2. Predecir posición de todos los tracks
    for (auto& track : tracks) {
        track.predict();
    }
    
    // 3. Primera asociación: High confidence detections
    std::vector<int> all_track_indices(tracks.size());
    std::iota(all_track_indices.begin(), all_track_indices.end(), 0);
    
    std::vector<std::pair<int, int>> matches;
    std::vector<int> unmatched_dets, unmatched_tracks;
    
    associate_detections_to_tracks(high_conf, all_track_indices,
                                   matches, unmatched_dets, unmatched_tracks);
    
    // 4. Actualizar tracks matcheados
    for (const auto& [det_idx, track_idx] : matches) {
        tracks[track_idx].update(high_conf[det_idx].box);
        tracks[track_idx].confidence = high_conf[det_idx].confidence;
        
        // Copiar landmarks
        for (int i = 0; i < 5; i++) {
            tracks[track_idx].landmarks[i] = high_conf[det_idx].landmarks[i];
        }
        
        tracks[track_idx].hits++;
        tracks[track_idx].time_since_update = 0;
        tracks[track_idx].age++;
    }
    
    // 5. Segunda asociación: Low confidence con tracks no matcheados
    if (!low_conf.empty() && !unmatched_tracks.empty()) {
        std::vector<std::pair<int, int>> matches2;
        std::vector<int> unmatched_dets2, unmatched_tracks2;
        
        associate_detections_to_tracks(low_conf, unmatched_tracks,
                                      matches2, unmatched_dets2, unmatched_tracks2);
        
        for (const auto& [det_idx, track_idx] : matches2) {
            tracks[track_idx].update(low_conf[det_idx].box);
            tracks[track_idx].confidence = low_conf[det_idx].confidence;
            
            for (int i = 0; i < 5; i++) {
                tracks[track_idx].landmarks[i] = low_conf[det_idx].landmarks[i];
            }
            
            tracks[track_idx].hits++;
            tracks[track_idx].time_since_update = 0;
            tracks[track_idx].age++;
        }
        
        unmatched_tracks = unmatched_tracks2;
    }
    
    // 6. Crear nuevos tracks para detecciones high conf sin match
    for (int det_idx : unmatched_dets) {
        TrackedFace new_track;
        new_track.id = next_id++;
        new_track.box = high_conf[det_idx].box;
        new_track.confidence = high_conf[det_idx].confidence;
        
        for (int i = 0; i < 5; i++) {
            new_track.landmarks[i] = high_conf[det_idx].landmarks[i];
        }
        
        new_track.age = 1;
        new_track.hits = 1;
        new_track.time_since_update = 0;
        new_track.init_kalman();
        
        tracks.push_back(new_track);
    }
    
    // 7. Incrementar time_since_update para tracks no matcheados
    for (int track_idx : unmatched_tracks) {
        tracks[track_idx].time_since_update++;
        tracks[track_idx].age++;
    }
    
    // 8. Eliminar tracks antiguos
    tracks.erase(
        std::remove_if(tracks.begin(), tracks.end(),
            [this](const TrackedFace& t) {
                return t.time_since_update > max_age;
            }),
        tracks.end()
    );
    
    // 9. Retornar tracks confirmados
    return get_active_tracks();
}

std::vector<TrackedFace> FaceTracker::get_active_tracks() const {
    std::vector<TrackedFace> active;
    for (const auto& track : tracks) {
        if (track.hits >= min_hits && track.time_since_update == 0) {
            active.push_back(track);
        }
    }
    return active;
}

int FaceTracker::get_active_count() const {
    return get_active_tracks().size();
}