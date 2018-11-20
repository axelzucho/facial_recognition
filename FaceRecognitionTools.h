//
// Created by axelzucho on 16/11/18.
//

#ifndef FACIAL_RECOGNITION_FACERECOGNITIONTOOLS_H
#define FACIAL_RECOGNITION_FACERECOGNITIONTOOLS_H

#include "opencv2/opencv.hpp"
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>

//Initializes struct used for Face Detection
parameters_FacDet initialize_detection_parameters();

//General function to show the case 1 results
void show_case_1(const Mat &image_taken, const string &matricula, std::pair<int, BiographicalData> result_case_1);

//Specific function to show the case 1 result when there is a match in the DB
void show_case_1_match(const Mat &image_taken, const Mat &image_db, const BiographicalData &data);

//Specific function to show the case 1 result when there is no match in the DB
void show_case_1_no_match(const Mat &image_taken, const Mat &image_db);

//Specific function to show the case 1 result when the ID isn't registered in the DB
void show_case_1_no_information(const Mat &image_taken, const string &matricula);

//General function to show if case 2 has an error
void show_case_2E(cv::Mat image);

//General function to show the case 2 results
void show_case_2(cv::Mat image_taken, cv::Mat image_db,
                 std::pair<int, std::vector<std::pair<BiographicalData, float>>> result_case_2);

//General function to show the case 3 results
void show_case_3(const Mat &image_taken, const BiographicalData &data_added);

//Adds text that displays the options for the interface
void add_options_to_image(Mat &image);

//Used for the I/O in a given openCV image. The input will be read till the 'Intro' key is pressed
string get_input_from_image(const Mat &image, string output_to_user);

//Adds text to the image
void show_text_in_image(const Mat &image, const std::string text);

//Used to control the input taken from the openCV image
void add_valid_char(std::string &text, const char char_to_add);

//Displays the image and the points given for confirmation
void show_image_confirmation(const Mat &image, const dlib::full_object_detection &shape);

#endif //FACIAL_RECOGNITION_FACERECOGNITIONTOOLS_H
