//
// Created by axelzucho on 16/11/18.
//

#include "FaceRecognition.h"
#include "opencv2/opencv.hpp"

#include <iostream>
#include <vector>

#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>

parameters_FacDet initialize_detection_parameters() {
    parameters_FacDet settings;//for detector initialization
    std::vector<string> paths_to_detection_models;

    paths_to_detection_models.push_back("../FaceDetection/classifiers/haarcascade_frontalface_default.xml");
    paths_to_detection_models.push_back("../FaceDetection/classifiers/haarcascade_frontalface_alt.xml");
    paths_to_detection_models.push_back("../FaceDetection/classifiers/haarcascade_frontalface_alt2.xml");
    paths_to_detection_models.push_back("../FaceDetection/classifiers/haarcascade_frontalface_alt_tree.xml");

    settings.classifiers_location = paths_to_detection_models;
    settings.scaleFact = 1.2;
    settings.validNeighbors = 1;
    settings.minWidth = 30;
    settings.maxWidth = 30;

    return settings;
}

void show_case_1_match(const Mat &image_taken, const Mat &image_db, const BiographicalData &data) {
    cv::Mat frame(cv::Size(image_taken.cols + image_db.cols + 100, image_taken.rows + 200), image_taken.type(),
                  cv::Scalar(0));
    //Frame for the image taken
    cv::Mat image_taken_frame(frame, cv::Rect(20, 20, image_taken.cols, image_taken.rows));

    //Frame for the image in the DB
    cv::Mat image_db_frame(frame, cv::Rect(image_taken.cols + 60, 20, image_db.cols, image_db.rows));

    image_taken.copyTo(image_taken_frame);
    image_db.copyTo(image_db_frame);
    cv::putText(frame, "Persona verificada correctamente", cv::Point(20, image_taken.rows + 40), cv::FONT_HERSHEY_PLAIN,
                1.5, cv::Scalar(255, 255, 255));
    cv::putText(frame, "Nombre: " + data.name, cv::Point(20, image_taken.rows + 80), cv::FONT_HERSHEY_PLAIN, 1.5,
                cv::Scalar(255, 255, 255));
    cv::putText(frame, "Apellido: " + data.lastName, cv::Point(20, image_taken.rows + 110), cv::FONT_HERSHEY_PLAIN, 1.5,
                cv::Scalar(255, 255, 255));
    cv::putText(frame, "Matricula: " + data.matricula, cv::Point(20, image_taken.rows + 140), cv::FONT_HERSHEY_PLAIN,
                1.5, cv::Scalar(255, 255, 255));
    cv::putText(frame, "Mail: " + data.mail, cv::Point(20, image_taken.rows + 170), cv::FONT_HERSHEY_PLAIN, 1.5,
                cv::Scalar(255, 255, 255));


    cv::imshow("Output case 1", frame);
    cv::waitKey(0);
}

void show_case_1_no_match(const Mat &image_taken, const Mat &image_db) {
    cv::Mat frame(cv::Size(image_taken.cols + image_db.cols + 100, image_taken.rows + 200), image_taken.type(),
                  cv::Scalar(0));
    //Frame for the image taken
    cv::Mat image_taken_frame(frame, cv::Rect(20, 20, image_taken.cols, image_taken.rows));

    //Frame for the image in the DB
    cv::Mat image_db_frame(frame, cv::Rect(image_taken.cols + 60, 20, image_db.cols, image_db.rows));

    image_taken.copyTo(image_taken_frame);
    image_db.copyTo(image_db_frame);

    cv::putText(frame, "La persona NO es la misma que", cv::Point(20, image_taken.rows + 80), cv::FONT_HERSHEY_PLAIN, 2,
                cv::Scalar(255, 255, 255));
    cv::putText(frame, "la matricula ingresada", cv::Point(20, image_taken.rows + 110), cv::FONT_HERSHEY_PLAIN, 2,
                cv::Scalar(255, 255, 255));

    cv::imshow("Output case 1", frame);
    cv::waitKey(0);
}

void show_case_1_no_information(const Mat &image_taken, const string &matricula) {
    cv::Mat frame(cv::Size(image_taken.cols * 2 + 100, image_taken.rows + 200), image_taken.type(), cv::Scalar(0));
    //Frame for the image taken
    cv::Mat image_taken_frame(frame, cv::Rect(20, 20, image_taken.cols, image_taken.rows));

    image_taken.copyTo(image_taken_frame);

    cv::putText(frame, "La matricula " + matricula, cv::Point(20, image_taken.rows + 80), cv::FONT_HERSHEY_PLAIN, 2,
                cv::Scalar(255, 255, 255));
    cv::putText(frame, "NO esta registrada", cv::Point(20, image_taken.rows + 110), cv::FONT_HERSHEY_PLAIN, 2,
                cv::Scalar(255, 255, 255));

    cv::imshow("Output case 1", frame);
    cv::waitKey(0);

}

//The actual return numbers should be replaced once available
void show_case_1(const Mat &image_taken, const string &matricula, std::pair<int, BiographicalData> result_case_1) {
    if (result_case_1.first == 1) {
        //This should be replaced with the aligned DB image once available
        Mat image_db = image_taken.clone();
        show_case_1_match(image_taken, image_db, result_case_1.second);
    } else if (result_case_1.first == 0) {
        //This should be replaced with the aligned DB image once available
        Mat image_db = image_taken.clone();
        show_case_1_no_match(image_taken, image_db);
    } else if (result_case_1.first == -2) {
        show_case_1_no_information(image_taken, matricula);
    }
}

void show_case_2E(cv::Mat image)
{
    cv::Mat frames(cv::Size(700, 60), image.type(), cv::Scalar(0));
    cv::putText(frames, "Error al identificar la persona, intente de nuevo", cv::Point(20, 30),
                cv::FONT_HERSHEY_PLAIN, 1.5, cv::Scalar(255, 255, 255));
    cv::imshow("Output case 2 (ERROR)", frames);
    cv::waitKey(0);
}

void show_case_2(cv::Mat image_taken, cv::Mat image_db,
                 std::pair<int, std::vector<std::pair<BiographicalData, float>>> result_case_2) {
    int vector_display_size = 5;

    cv::Mat frame(cv::Size(image_taken.cols + 250, image_taken.rows + 80), image_taken.type(), cv::Scalar(0));
//cv::resize(image_db, image_db, cv::Size(), 0.25, 0.25, cv::INTER_CUBIC);
    cv::resize(image_taken, image_taken, cv::Size(), 0.75, 0.75, cv::INTER_CUBIC);
//Frame for the image taken
    cv::Mat image_taken_frame(frame, cv::Rect(20, 20, image_taken.cols, image_taken.rows));

//Frame for the image in the DB
//cv::Mat image_db_frame(frame, cv::Rect(20, image_taken.rows + 30, image_db.cols, image_db.rows));

    image_taken.copyTo(image_taken_frame);
//image_db.copyTo(image_db_frame);
//cv::putText(frame, "40%", cv::Point(20, image_taken.rows + image_db.rows + 30), cv::FONT_HERSHEY_PLAIN, 1.5, cv::Scalar(140, 244, 66));

    if (result_case_2.second.size() < vector_display_size) {
        vector_display_size = result_case_2.second.size();
    }
    for (int i = 0; i < vector_display_size; i++) {
        cv::Mat recognized_image;
        recognized_image = cv::imread(result_case_2.second.at(i).first.img, cv::IMREAD_COLOR);

        cv::resize(recognized_image, image_db, cv::Size(), 0.20, 0.20, cv::INTER_CUBIC);
        cv::Mat image_db_frame(frame, cv::Rect(20 + 20 * i + image_db.cols * i, image_taken.rows + 30, image_db.cols,
                                               image_db.rows));
        image_db.copyTo(image_db_frame);
        cv::putText(frame, std::to_string(result_case_2.second.at(i).second),
                    cv::Point(20 + 20 * i + image_db.cols * i, image_taken.rows + image_db.rows + 30),
                    cv::FONT_HERSHEY_PLAIN, 1.5, cv::Scalar(140, 244, 66));
    }
    cv::putText(frame, "Persona identificada: ", cv::Point(image_taken.cols + 40, 40), cv::FONT_HERSHEY_PLAIN, 1.5,
                cv::Scalar(255, 255, 255));
    cv::putText(frame, "Nombre: " + result_case_2.second.front().first.name, cv::Point(image_taken.cols + 40, 80),
                cv::FONT_HERSHEY_PLAIN, 1.5, cv::Scalar(255, 255, 255));
    cv::putText(frame, "Apellido: " + result_case_2.second.front().first.lastName,
                cv::Point(image_taken.cols + 40, 110), cv::FONT_HERSHEY_PLAIN, 1.5, cv::Scalar(255, 255, 255));
    cv::putText(frame, "Matricula: " + result_case_2.second.front().first.matricula,
                cv::Point(image_taken.cols + 40, 140), cv::FONT_HERSHEY_PLAIN, 1.5, cv::Scalar(255, 255, 255));
    cv::putText(frame, "Mail: " + result_case_2.second.front().first.mail, cv::Point(image_taken.cols + 40, 170),
                cv::FONT_HERSHEY_PLAIN, 1.5, cv::Scalar(255, 255, 255));
    cv::imshow("Output case 2", frame);

    cv::waitKey(0);
}

void show_case_3(const Mat &image_taken, const BiographicalData &result_case_3) {
    cv::Mat frame(cv::Size(image_taken.cols + 400, image_taken.rows + 80), image_taken.type(), cv::Scalar(0));
    cv::Mat image_taken_frame(frame, cv::Rect(20, 20, image_taken.cols, image_taken.rows));
    image_taken.copyTo(image_taken_frame);
    cv::putText(frame, "Datos ingresados: ", cv::Point(image_taken.cols + 40, 40), cv::FONT_HERSHEY_PLAIN, 1.5,
                cv::Scalar(255, 255, 255));
    cv::putText(frame, "Nombre: " + result_case_3.name, cv::Point(image_taken.cols + 40, 80), cv::FONT_HERSHEY_PLAIN,
                1.5, cv::Scalar(255, 255, 255));
    cv::putText(frame, "Apellido: " + result_case_3.lastName, cv::Point(image_taken.cols + 40, 110),
                cv::FONT_HERSHEY_PLAIN, 1.5, cv::Scalar(255, 255, 255));
    cv::putText(frame, "Matricula: " + result_case_3.matricula, cv::Point(image_taken.cols + 40, 140),
                cv::FONT_HERSHEY_PLAIN, 1.5, cv::Scalar(255, 255, 255));
    cv::putText(frame, "Mail: " + result_case_3.mail, cv::Point(image_taken.cols + 40, 170), cv::FONT_HERSHEY_PLAIN,
                1.5, cv::Scalar(255, 255, 255));
    cv::putText(frame, "Edad: " + std::to_string(result_case_3.age), cv::Point(image_taken.cols + 40, 200),
                cv::FONT_HERSHEY_PLAIN, 1.5, cv::Scalar(255, 255, 255));
    cv::putText(frame, "Persona enrolada correctamente a la base de datos: ", cv::Point(20, 40 + image_taken.rows),
                cv::FONT_HERSHEY_PLAIN, 1.5, cv::Scalar(255, 255, 255));
    cv::imshow("Output case 3", frame);
    cv::waitKey(0);
}

void add_options_to_image(Mat &image) {
    std::string first_line = "1. Verificar con matricula";
    std::string second_line = "2. Reconocer";
    std::string third_line = "3. Enrolar";
    cv::putText(image, first_line, cv::Point(20, image.rows - 100), cv::FONT_HERSHEY_PLAIN, 2,
                cv::Scalar(255, 255, 255), 4);
    cv::putText(image, second_line, cv::Point(20, image.rows - 60), cv::FONT_HERSHEY_PLAIN, 2,
                cv::Scalar(255, 255, 255), 4);
    cv::putText(image, third_line, cv::Point(20, image.rows - 20), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(255, 255, 255),
                4);
}

void show_text_in_image(const Mat &image, const std::string text) {
    //cv::destroyAllWindows();
    cv::Mat image_to_show = image.clone();
    cv::putText(image_to_show, text, cv::Point(10, image.rows - 30), cv::FONT_HERSHEY_PLAIN, 1,
                cv::Scalar(255, 255, 255));
    cv::imshow("Interface", image_to_show);
}

void add_valid_char(std::string &text, const char char_to_add) {
    //std::cout << (int) char_to_add << std::endl;
    if ((char_to_add >= 'A' && char_to_add <= 'Z') || (char_to_add >= '0' && char_to_add <= '9')) {
        text += char_to_add;
        return;
    }
    if (char_to_add >= 'a' && char_to_add <= 'z'){
        text += char_to_add - 32;
        return;
    }
    if (char_to_add >= 'a' && char_to_add <= 'z'){
        text += char_to_add - 32;
        return;
    }
    if (char_to_add == '@' || char_to_add == '.' || char_to_add == ' ') {
        text += char_to_add;
        return;
    }
    if (char_to_add == 8 && !text.empty()) {
        text.pop_back();
        return;
    }
    return;
}

string get_input_from_image(const Mat &image, string output_to_user) {
    string user_input = "";
    char case_key_pressed = cv::waitKey(0);
    while (case_key_pressed != '\n' && case_key_pressed != 13) {
        add_valid_char(user_input, case_key_pressed);
        string text_to_show = output_to_user;
        int user_input_beg = user_input.length() >= 36 - output_to_user.length() ? user_input.length() - 36 + output_to_user.length() : 0;
        text_to_show += user_input.substr(user_input_beg, user_input.length());
        show_text_in_image(image, text_to_show);
        case_key_pressed = cv::waitKey(0);
    }
    cv::destroyAllWindows();
    return user_input;
}

void show_image_confirmation(const Mat &image, const dlib::full_object_detection &shape) {
    Mat image_to_show = image.clone();
    for (int i = 0; i < shape.num_parts(); ++i) {
        cv::circle(image_to_show, cv::Point(shape.part(i).x(), shape.part(i).y()), 3, cv::Scalar(255, 255, 255));
    }
    cv::putText(image_to_show, "Estan correctamente distribuidos los puntos en la cara?",
                cv::Point(20, image_to_show.rows - 100), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 255, 255), 2);
    cv::putText(image_to_show, "(S)Si", cv::Point(image_to_show.cols / 2 - 50, image_to_show.rows - 50),
                cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 255, 255), 2);
    cv::putText(image_to_show, "(N)No", cv::Point(image_to_show.cols / 2 + 50, image_to_show.rows - 50),
                cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 255, 255), 2);
    cv::imshow("Confirmation", image_to_show);
}
