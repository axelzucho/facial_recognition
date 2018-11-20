//
// Created by axelzucho on 30/10/18.
//

#include "FaceRecognition.h"
#include "opencv2/opencv.hpp"
#include "FaceRecognitionTools.h"

#include <iostream>
#include <vector>

int main() {
    parameters_FacDet settings = initialize_detection_parameters();

    //Some initialization for the first raw example...
    std::vector<cv::Rect> all_faces;
    std::vector<cv::Rect> real_faces;
    std::vector<cv::Rect> largest_face;
    largest_face.resize(1);

    //path of classifiers to train algorithm
    FaceRecognition face_recognition(settings, "../FaceAligner/shape_predictor_5_face_landmarks.dat", 150, 0.3,
                                     "../FaceDescriptorExtractor/dlib_face_recognition_resnet_model_v1.dat", 0.5, 5);

    //Used to display the images at the end of each case
    FaceAligner interface_face_aligner("../FaceAligner/shape_predictor_5_face_landmarks.dat", 350, 0.1);
    Mat interface_image;


    //Create database object
    cv::Mat template_image;
    dlib::full_object_detection shape;

    bool run_program = true;
    while (run_program) {
        cv::destroyAllWindows();
        cv::VideoCapture video(0);
        video.open(0);

        while (!video.isOpened()) {
            std::cout << "Error al intentar abrir la camara" << std::endl;
        }

        bool flag = true;
        while (flag) {
            cv::Mat frame;
            video >> frame;

            all_faces = face_recognition.face_detector_->detect_faces(&frame);
            real_faces = face_recognition.face_detector_->ignore_false_positives(&frame, all_faces, 2);
            largest_face[0] = face_recognition.face_detector_->get_largest_face(real_faces);

            Mat frame_with_rectangles = frame.clone();
            add_options_to_image(frame_with_rectangles);

            face_recognition.face_detector_->show_faces(&frame_with_rectangles, all_faces, real_faces, largest_face[0]);
            face_recognition.face_aligner_->Detect(frame, largest_face[0], shape);

            char key_pressed = cv::waitKey(1);
            if (key_pressed == '1' || key_pressed == '2' || key_pressed == '3') {
                if (!frame.empty()) {
                    cv::destroyAllWindows();
                    show_image_confirmation(frame, shape);

                    char confirmation_option = 'a';
                    while (confirmation_option != 's' && confirmation_option != 'S' && confirmation_option != 'n' &&
                           confirmation_option != 'N') {
                        confirmation_option = cv::waitKey(0);
                    }
                    cv::destroyAllWindows();
                    if (confirmation_option != 's' && confirmation_option != 'S') continue;
                } else continue;
            }

            switch (key_pressed) {
                case '1':
                    //Example for showing case 1, should be correctly implemented at the end when the shape is stored.
                    //interface_face_aligner.Align(shape, frame, interface_image);
                    //show_case_1(interface_image, interface_image, "La persona fue reconocida con exito");
                    std::cout << "1";
                    cv::destroyAllWindows();
                    video.release();
                    std::cout << "Ingrese la matricula" << "\n";
                    // Aqui se implementa caso 1
                    {
                        interface_face_aligner.Align(shape, frame, interface_image);
                        std::string text_to_show = "Ingrese la matricula: ";
                        std::string matricula = get_input_from_image(interface_image, text_to_show);
                        std::pair<int, BiographicalData> result_case_1;
                        result_case_1 = face_recognition.caso1(&frame, shape, matricula);
                        show_case_1(interface_image, matricula, result_case_1);
                    }
                    flag = false;
                    break;

                case '2':
                    std::cout << "2";
                    cv::destroyAllWindows();
                    video.release();
                    // Aqui se implementa caso 2

                    if (!frame.empty()) {
                        std::pair<int, std::vector<std::pair<BiographicalData, float>>> result_case_2;
                        result_case_2 = face_recognition.caso2(&frame, shape);


                        if (result_case_2.first == 1) {
                            cv::Mat recognized_image;
                            recognized_image = cv::imread(result_case_2.second.at(0).first.img, cv::IMREAD_COLOR);
                            show_case_2(frame, recognized_image, result_case_2);
                            
                            //std::cout << "La persona fue reconocida en la base de datos como: " << result_case_2.second.front().first.name << " " << result_case_2.second.front().first.lastName << " con la matrícula: " << result_case_2.second.front().first.matricula << "\n";
                        } else {
                            //std::cout << "La persona no fue reconocida\n";
                        	show_case_2E(frame);
                        }
                    } else {
                        flag = true;
                        break;
                    }

                    flag = false;
                    break;

                case '3':
                    std::cout << "1";
                    cv::destroyAllWindows();
                    video.release();
                    std::cout << "Ingrese los datos a escribir" << "\n";
                    // Aqui se implementa caso 3
                    {
                        BiographicalData bio;
                        interface_face_aligner.Align(shape, frame, interface_image);
                        string matricula, name, last_name, mail, text_to_show;
                        int age, result_case_3 = -31;
                        //Errors defined in DataBase.cpp
                        while (result_case_3 != 1) {
                            result_case_3 *= -1;
                            if (result_case_3 & 1) {
                                text_to_show = "Ingrese la matricula: ";
                                matricula = get_input_from_image(interface_image, text_to_show);
                            }
                            if (result_case_3 & 2) {
                                text_to_show = "Ingrese el nombre: ";
                                name = get_input_from_image(interface_image, text_to_show);
                            }
                            if (result_case_3 & 4) {
                                text_to_show = "Ingrese el apellido: ";
                                last_name = get_input_from_image(interface_image, text_to_show);
                            }
                            if (result_case_3 & 8) {
                                text_to_show = "Ingrese el mail: ";
                                mail = get_input_from_image(interface_image, text_to_show);
                            }
                            if (result_case_3 & 16) {
                                text_to_show = "Ingrese la edad: ";
                                try {
                                    age = std::stoi(get_input_from_image(interface_image, text_to_show));
                                }
                                catch (std::invalid_argument &e) {
                                    age = -1;
                                }
                            }
                            if (result_case_3 & 32) {
                                std::cout << "La matrícula ya está registrada, intente de nuevo." << std::endl;
                                break;
                            }

                            bio.matricula = matricula;
                            bio.age = age;
                            bio.name = name;
                            bio.lastName = last_name;
                            bio.mail = mail;
                            result_case_3 = face_recognition.enroll(frame, shape, bio);
                            std::cout << result_case_3 << std::endl;
                        }
                        if (result_case_3 == 1) {
                            //std::cout << "La persona fue registrada\n";
                            show_case_3(frame, bio);
                        }

                    }
                    flag = false;
                    break;

                case '0':
                    cv::destroyAllWindows();
                    video.release();
                    run_program = false;
                    flag = false;
                    break;

                default:
                    break;
            }

        }
    }

    return 0;
}
