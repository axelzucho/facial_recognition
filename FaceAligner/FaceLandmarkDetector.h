#ifndef FACIAL_TRANSFORMATION_FACEDETECTOR_H
#define FACIAL_TRANSFORMATION_FACEDETECTOR_H

#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <opencv2/core/core.hpp>

using std::string;

class FaceLandmarkDetector {
public:
    FaceLandmarkDetector(const string &path_to_model);

    void GetFaceLandmark(const dlib::array2d<dlib::bgr_pixel> &image, const dlib::rectangle &face,
                         dlib::full_object_detection &shape);

    void GetFaceLandmark(const cv::Mat &image, const dlib::rectangle &face,
                         dlib::full_object_detection &shape);

private:
    dlib::shape_predictor sp;
};

#endif //FACIAL_TRANSFORMATION_FACEDETECTOR_H
