#include "FaceAligner.h"
#include "FaceLandmarkDetector.h"
#include "Aligner.h"
#include "facial_extractor_tools.h"

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/opencv.h>

FaceAligner::FaceAligner(){
    face_landmark_detector_= new FaceLandmarkDetector("../shape_predictor_5_face_landmarks.dat");
    aligner_ = new Aligner(150, 0.2);
}

FaceAligner::FaceAligner(const string& path_to_model, const unsigned int size, const double left_eye_after){
    face_landmark_detector_ = new FaceLandmarkDetector(path_to_model);
    aligner_ = new Aligner(size, left_eye_after);
}

void FaceAligner::DetectAndAlign(const cv::Mat &image, const cv::Rect &face, cv::Mat& template_image){
    CV_Assert(!face.empty());
    CV_Assert(!image.empty());
    dlib::full_object_detection shape;
    dlib::array2d<dlib::bgr_pixel> dlib_image;
    assign_image(dlib_image, dlib::cv_image<dlib::bgr_pixel>(image));

    face_landmark_detector_->GetFaceLandmark(dlib_image, extractor::opencv_rect_to_dlib(face), shape);
    aligner_->AlignImage(shape, dlib_image, template_image);
}

void FaceAligner::Detect(const cv::Mat &image, const cv::Rect &face, dlib::full_object_detection &shape) {
    face_landmark_detector_->GetFaceLandmark(image, extractor::opencv_rect_to_dlib(face), shape);
}

void FaceAligner::Align(const dlib::full_object_detection &shape, const cv::Mat &image, cv::Mat &template_image){
    aligner_->AlignImage(shape,image,template_image);
}

FaceAligner::~FaceAligner() {
    delete(face_landmark_detector_);
    delete(aligner_);
}

//
// Created by axelzucho on 15/10/18.
//

