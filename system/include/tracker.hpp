// ============= include/tracker.hpp =============
/*
 * Face Tracking con ByteTrack Algorithm
 * 
 * CARACTERÍSTICAS:
 * - Tracking multi-objeto basado en IoU
 * - Kalman filter para predicción de movimiento
 * - Two-stage association (high/low confidence)
 * - ID persistente durante oclusiones temporales
 * 
 * PARÁMETROS:
 * - iou_threshold: umbral para asociar detecciones (0.3 default)
 * - max_age: frames sin detección antes de eliminar (30 = 1 seg)
 * - min_hits: detecciones necesarias para confirmar track (3)
 * 
 * PERFORMANCE:
 * - Overhead: ~2ms por frame
 * - Memory: ~200 bytes por track
 * 
 * AUTOR: PANTO System
 * FECHA: 2025
 */

#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>
#include <vector>
#include <string>

struct Detection;

struct TrackedFace {
    int id;
    cv::Rect box;
    cv::Point2f landmarks[5];
    float confidence;
    
    int age;
    int hits;
    int time_since_update;
    
    cv::KalmanFilter kf;
    bool kf_initialized;
    
    bool is_recognized;
    std::string name;
    
    std::string emotion;
    float emotion_confidence;
    
    int age_years;
    std::string gender;
    float gender_confidence;
    
    // ===== NUEVO: Para DB Manager =====
    std::vector<float> embedding;
    float best_quality;
    std::string person_id;
    // ==================================
    
    TrackedFace();
    void init_kalman();
    void predict();
    void update(const cv::Rect& measurement);
};

class FaceTracker {
public:
    FaceTracker(float iou_threshold = 0.3f, 
                int max_age = 30,
                int min_hits = 3);
    
    std::vector<TrackedFace> update(const std::vector<Detection>& detections);
    std::vector<TrackedFace> get_active_tracks() const;
    
    int get_total_tracks() const { return next_id - 1; }
    int get_active_count() const;
    
private:
    int next_id;
    std::vector<TrackedFace> tracks;
    
    float iou_threshold;
    int max_age;
    int min_hits;
    
    float calculate_iou(const cv::Rect& a, const cv::Rect& b);
    
    void associate_detections_to_tracks(
        const std::vector<Detection>& detections,
        const std::vector<int>& track_indices,
        std::vector<std::pair<int, int>>& matches,
        std::vector<int>& unmatched_dets,
        std::vector<int>& unmatched_tracks
    );
    
    void linear_assignment(
        const std::vector<std::vector<float>>& iou_matrix,
        const std::vector<int>& detection_indices,
        const std::vector<int>& track_indices,
        std::vector<std::pair<int, int>>& matches,
        std::vector<int>& unmatched_dets,
        std::vector<int>& unmatched_tracks
    );
};