#ifndef FACIAL_TRANSFORMATION_Aligner_H
#define FACIAL_TRANSFORMATION_Aligner_H

#include <dlib/image_processing/frontal_face_detector.h>
#include <opencv2/opencv.hpp>

class Aligner {
public:
    Aligner(const unsigned int size, const double left_eye_after);
    void AlignImage(const dlib::full_object_detection &shape, const dlib::array2d<dlib::bgr_pixel>& image, cv::Mat &template_image);
    void AlignImage(const dlib::full_object_detection &shape, const cv::Mat &image, cv::Mat &template_image);

private:
    unsigned int size_;
    double left_eye_after_;
};

#endif //FACIAL_TRANSFORMATION_Aligner_H
