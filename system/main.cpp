#include "config_loader.hpp"
#include "detector.hpp"
#include "tracker.hpp"
#include "sr_module.hpp"
#include "recognizer.hpp"
#include "database.hpp"

int main() {
    Config cfg("config/config.yaml");
    Detector detector(cfg.model.detector);
    Recognizer recognizer(cfg.model.recognizer);
    SRModule sr(cfg.model.sr_gfpgan, cfg.model.sr_realesrgan);
    Database db(cfg.database.path);
    Tracker tracker;

    cv::VideoCapture cap(cfg.video.source);
    if (!cap.isOpened()) return -1;

    while (true) {
        cv::Mat frame;
        if (!cap.read(frame)) break;

        auto faces = detector.detect(frame);
        tracker.update(faces);

        for (auto& f : tracker.active_faces()) {
            cv::Mat faceROI = detector.extract_face(frame, f.bbox);
            if (cfg.pipeline.use_superres && f.size < cfg.pipeline.min_face_size)
                faceROI = sr.enhance(faceROI);

            auto emb = recognizer.embed(faceROI);
            int id = db.match(emb, cfg.database.threshold);
            db.log(id, emb);
        }
    }

    return 0;
}
