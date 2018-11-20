#include "Aligner.h"
#include "facial_extractor_tools.h"

#include <dlib/image_io.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_transforms.h>
#include <dlib/array2d.h>
#include <dlib/pixel.h>
#include <dlib/opencv.h>
#include <opencv2/core/core.hpp>

#include "facial_extractor_tools.h"

Aligner::Aligner(const unsigned int size, const double left_eye_after) :
        size_(size), left_eye_after_(left_eye_after) {}

void Aligner::AlignImage(const dlib::full_object_detection &shape, const dlib::array2d <dlib::bgr_pixel> &image,
                             cv::Mat &template_image) {
    dlib::array2d <dlib::bgr_pixel> face_chip;
    dlib::extract_image_chip(image, dlib::get_face_chip_details(shape, size_, left_eye_after_), face_chip);

    template_image = dlib::toMat(face_chip).clone();
}

void Aligner::AlignImage(const dlib::full_object_detection &shape, const cv::Mat &image, cv::Mat &template_image) {
    CV_Assert(!image.empty());
    dlib::array2d<dlib::bgr_pixel> dlib_image;
    assign_image(dlib_image, dlib::cv_image<dlib::bgr_pixel>(image));

    AlignImage(shape, dlib_image, template_image);
}

void Aligner::AlignImage(const std::vector<cv::Point2f> &points, const cv::Mat &image, cv::Mat &template_image) {
    std::vector<dlib::point> dlib_points;
    std::cout << "Puntos: " << points.size() << std::endl;
    for(auto it:points){
        dlib_points.push_back(dlib::point(it.x, it.y));
    }
    dlib::rectangle rect(points[0].x-10, points[0].y-50, points[0].x+140, points[0].y+100);
    dlib::full_object_detection shape(rect, dlib_points);
    CV_Assert(!image.empty());
    dlib::array2d<dlib::bgr_pixel> dlib_image;
    assign_image(dlib_image, dlib::cv_image<dlib::bgr_pixel>(image));

    AlignImage(shape, dlib_image, template_image);
}