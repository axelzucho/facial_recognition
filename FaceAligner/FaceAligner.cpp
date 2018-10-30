#include "FaceAligner.h"
#include "FaceLandmarkDetector.h"
#include "Aligner.h"
#include "facial_extractor_tools.h"

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/opencv.h>

FaceAligner::FaceAligner(const string& path_to_model, const unsigned int size, const double left_eye_after){
    face_detector_ = new FaceLandmarkDetector(path_to_model);
    aligner_ = new Aligner(size, left_eye_after);
}

void FaceAligner::DetectAndAlign(const cv::Mat &image, const cv::Rect &face, cv::Mat& template_image){
    dlib::full_object_detection shape;
    dlib::array2d<dlib::bgr_pixel> dlib_image;
    assign_image(dlib_image, dlib::cv_image<dlib::bgr_pixel>(image));

    face_detector_->GetFaceLandmark(dlib_image, extractor::opencv_rect_to_dlib(face), shape);
    aligner_->AlignImage(shape, dlib_image, template_image);
}

FaceAligner::~FaceAligner() {
    delete(face_detector_);
    delete(aligner_);
}

//
// Created by axelzucho on 15/10/18.
//

