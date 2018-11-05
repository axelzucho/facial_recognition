//
// Created by axelzucho on 15/10/18.
//

#ifndef FACIAL_TRANSFORMATION_FACETRANSFORMER_H
#define FACIAL_TRANSFORMATION_FACETRANSFORMER_H

#include "Aligner.h"
#include "FaceLandmarkDetector.h"

#include <dlib/image_processing.h>
#include <opencv2/core/core.hpp>

class FaceAligner {
public:
    FaceAligner();
    FaceAligner(const string& path_to_model, const unsigned int size, const double left_eye_after);
    void DetectAndAlign(const cv::Mat &image, const cv::Rect &face, cv::Mat &template_image);
    void Detect(const cv::Mat &image, const cv::Rect &face, dlib::full_object_detection &shape);
    void Align(const dlib::full_object_detection &shape, const cv::Mat &image, cv::Mat &template_image);
    ~FaceAligner();
private:
    FaceLandmarkDetector* face_landmark_detector_;
    Aligner* aligner_;
};

#endif //FACIAL_TRANSFORMATION_FACETRANSFORMER_H
