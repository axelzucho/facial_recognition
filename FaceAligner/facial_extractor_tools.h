#ifndef FACIAL_EXTRACTOR_TOOLS_H
#define FACIAL_EXTRACTOR_TOOLS_H

#include <opencv2/opencv.hpp>
#include <dlib/image_processing/frontal_face_detector.h>

namespace extractor {
    dlib::rectangle opencv_rect_to_dlib(cv::Rect r);

    dlib::point get_average(const dlib::point &left, const dlib::point &right);

    void show_image(const cv::Mat &image);

    void read_image(int argc, char **argv, std::string image_path, cv::Mat *image);

    } // namespace extractor

#endif