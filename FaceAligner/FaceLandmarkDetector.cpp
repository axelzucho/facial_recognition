#include "FaceLandmarkDetector.h"
#include "facial_extractor_tools.h"

#include <dlib/opencv/cv_image.h>

FaceLandmarkDetector::FaceLandmarkDetector(const string &path_to_model){
    dlib::deserialize(path_to_model) >> sp;
}

void FaceLandmarkDetector::GetFaceLandmark(const dlib::array2d<dlib::bgr_pixel> &image, const dlib::rectangle &face,
                                     dlib::full_object_detection &shape) {
    shape = sp(image, face);
}

void FaceLandmarkDetector::GetFaceLandmark(const cv::Mat &image, const dlib::rectangle &face,
                                           dlib::full_object_detection &shape) {
    CV_Assert(!image.empty());
    dlib::array2d<dlib::bgr_pixel> dlib_image;

    assign_image(dlib_image, dlib::cv_image<dlib::bgr_pixel>(image));

    GetFaceLandmark(dlib_image, face, shape);
}
