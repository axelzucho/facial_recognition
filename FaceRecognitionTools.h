//
// Created by axelzucho on 16/11/18.
//

#ifndef FACIAL_RECOGNITION_FACERECOGNITIONTOOLS_H
#define FACIAL_RECOGNITION_FACERECOGNITIONTOOLS_H

#include "opencv2/opencv.hpp"
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>

parameters_FacDet initialize_detection_parameters();
void show_case_1_match(const Mat& image_taken, const Mat& image_db, const BiographicalData& data);
void show_case_1_no_match(const Mat& image_taken, const Mat& image_db);
void show_case_1_no_information(const Mat& image_taken, const string& matricula);
void show_case_1(const Mat& image_taken, const string& matricula, std::pair<int, BiographicalData> result_case_1);
void show_case_2(cv::Mat image_taken, cv::Mat image_db, std::pair <int, std::vector<std::pair<BiographicalData, float>>> result_case_2);
void show_case_3(const Mat& image_taken, const BiographicalData& data_added);
void add_options_to_image(Mat& image);
void show_text_in_image(const Mat& image, const std::string text);
void add_valid_char(std::string &text, const char char_to_add);
void show_image_confirmation(const Mat &image, const dlib::full_object_detection &shape);
string get_input_from_image(const Mat &image, string output_to_user);


#endif //FACIAL_RECOGNITION_FACERECOGNITIONTOOLS_H
