//
// Created by axelzucho on 30/10/18.
//

#include "FaceRecognition.h"
#include "opencv2/opencv.hpp"

#include <iostream>
#include <vector>

parameters_FacDet initialize_detection_parameters(){
	parameters_FacDet settings;//for detector initialization
	std::vector <string> paths_to_detection_models;

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

void show_case_1(const Mat& image_now, const Mat& image_db, std::string text){
	cv::Mat frame(cv::Size(image_now.cols + image_db.cols + 100, image_now.rows + 200), image_now.type(), cv::Scalar(0));
	//Frame for the image taken
	cv::Mat image_now_frame(frame, cv::Rect(20, 20, image_now.cols, image_now.rows));

	//Frame for the image in the DB
	cv::Mat image_db_frame(frame, cv::Rect(image_now.cols + 60, 20, image_db.cols, image_db.rows));

    image_now.copyTo(image_now_frame);
    image_db.copyTo(image_db_frame);

    cv::putText(frame, text, cv::Point(20, image_now.rows + 50), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(255, 255, 255));

	cv::imshow("Output case 1", frame);
	cv::waitKey(0);
}

void show_case_2(cv::Mat image_taken, cv::Mat image_db, std::pair <int, std::vector<std::pair<BiographicalData, float>>> result_case_2){
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
    
    if (result_case_2.second.size() < vector_display_size)
    {
    	vector_display_size = result_case_2.second.size();
    }

    for (int i=0;i<vector_display_size;i++)
    {
    	cv::Mat recognized_image;
    	recognized_image = cv::imread(result_case_2.second.at(i).first.img, cv::IMREAD_COLOR);
    	cv::resize(recognized_image, image_db, cv::Size(), 0.25, 0.25, cv::INTER_CUBIC);
    	cv::Mat image_db_frame(frame, cv::Rect(20 + 20 * i + image_db.cols * i, image_taken.rows + 30, image_db.cols, image_db.rows));
    	image_db.copyTo(image_db_frame);
    	cv::putText(frame, std::to_string(result_case_2.second.at(i).second), cv::Point(20 + 20 * i + image_db.cols * i, image_taken.rows + image_db.rows + 30), cv::FONT_HERSHEY_PLAIN, 1.5, cv::Scalar(140, 244, 66));

    }
    cv::putText(frame, "Persona identificada: ", cv::Point(image_taken.cols + 40, 40), cv::FONT_HERSHEY_PLAIN, 1.5, cv::Scalar(255, 255, 255));
    cv::putText(frame, "Nombre: " + result_case_2.second.front().first.name, cv::Point(image_taken.cols + 40, 80), cv::FONT_HERSHEY_PLAIN, 1.5, cv::Scalar(255, 255, 255));
    cv::putText(frame, "Apellido: " + result_case_2.second.front().first.lastName, cv::Point(image_taken.cols + 40, 110), cv::FONT_HERSHEY_PLAIN, 1.5, cv::Scalar(255, 255, 255));
    cv::putText(frame, "Matricula: " + result_case_2.second.front().first.matricula, cv::Point(image_taken.cols + 40, 140), cv::FONT_HERSHEY_PLAIN, 1.5, cv::Scalar(255, 255, 255));
    cv::putText(frame, "Mail: " + result_case_2.second.front().first.mail, cv::Point(image_taken.cols + 40, 170), cv::FONT_HERSHEY_PLAIN, 1.5, cv::Scalar(255, 255, 255));
    cv::imshow("Output case 2", frame);
	cv::waitKey(0);
}

void show_case_3(const Mat& image_taken, const BiographicalData& data_added){
	return;
}

int main()
{
	parameters_FacDet settings = initialize_detection_parameters();

	//Some initialization for the first raw example...
	std::vector<cv::Rect> all_faces;
	std::vector<cv::Rect> real_faces;
  	std::vector<cv::Rect> largest_face;
  	largest_face.resize(1);

	//path of classifiers to train algorithm
	FaceRecognition face_recognition (settings, "../FaceAligner/shape_predictor_5_face_landmarks.dat", 150, 0.3, "../FaceDescriptorExtractor/dlib_face_recognition_resnet_model_v1.dat", 0.4, 5);

	//Used to display the images at the end of each case
	FaceAligner interface_face_aligner("../FaceAligner/shape_predictor_5_face_landmarks.dat", 350, 0.1);
    Mat interface_image;


	//Create database object
	cv::Mat template_image;
	dlib::full_object_detection shape;

	bool run_program = true;
	while(run_program)
	{
		cv::VideoCapture video(0);
		video.open(0);

		while (!video.isOpened()) {
		    std::cout << "Error al intentar abrir la camara" << std::endl;
		}

		std::cout << "Ingrese el número de la operación deseada:\n1. Reconocer una persona en camara\n2. Verificar si la persona ubicada coincide con la matricula ingresada\n3. Enrolar una persona nueva" << std::endl;
		bool flag = true;
		while (flag)
		{
			cv::Mat frame;
			video >> frame;
			all_faces = face_recognition.face_detector_->detect_faces(&frame);
			real_faces = face_recognition.face_detector_->ignore_false_positives(&frame, all_faces, 2);
			largest_face[0] = face_recognition.face_detector_->get_largest_face(real_faces);
			Mat frame_with_rectangles = frame.clone();
			face_recognition.face_detector_->show_faces(&frame_with_rectangles, all_faces, real_faces, largest_face[0]);
			face_recognition.face_aligner_->Detect(frame, largest_face[0], shape);
			char key_pressed = cv::waitKey(1);
			switch(key_pressed)
			{
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
						string matricula;
						std::cin >> matricula;
						std::pair<int, BiographicalData> result_case_1;
						result_case_1 = face_recognition.caso1(&frame, shape, matricula);

						if(result_case_1.first == 1){
							std::cout << "La persona concuerda con la matrícula ingresada\n";
							cv::Mat recognized_image;
							recognized_image = cv::imread(result_case_1.second.img, cv::IMREAD_COLOR);
							//show_case_1(frame, recognized_image);
							/*cv::resize(frame, frame, cv::Size(150, 150), 0, 0, cv::INTER_CUBIC);
							cv::hconcat(frame, recognized_image, recognized_image);
							cv::imshow("Recognized image vs Database Image",  recognized_image);
							cv::waitKey(0);*/

						}
						else {
							std::cout << "La persona NO concuerda con la matrícula ingresada\n";
						}
					}
	        	flag = false;
				break;

				case '2':
				std::cout << "2";
				cv::destroyAllWindows();
	        	video.release();
	        	// Aqui se implementa caso 2

				if(!frame.empty())
				{
					std::pair <int, std::vector<std::pair<BiographicalData, float>>> result_case_2;
					result_case_2 = face_recognition.caso2(&frame, shape);

					if(result_case_2.first == 1){
						cv::Mat recognized_image;
    					recognized_image = cv::imread(result_case_2.second.at(0).first.img, cv::IMREAD_COLOR);
    					show_case_2(frame, recognized_image, result_case_2);
						std::cout << "La persona fue reconocida en la base de datos como: " << result_case_2.second.front().first.name << " " << result_case_2.second.front().first.lastName << " con la matrícula: " << result_case_2.second.front().first.matricula << "\n";
					}
					else{
						std::cout << "La persona no fue reconocida\n";
					}
				}else{
					flag = true;
					break;
				}

	        	flag = false;
				break;

				case '3':
				std::cout << "1";
				cv::destroyAllWindows();
	        	video.release();
	        	std:: cout << "Ingrese los datos a escribir" << "\n";
	        	// Aqui se implementa caso 3
					{
						string matricula, name, last_name, mail;
						int age,result_case_3=-31;
						//Errors defined in DataBase.cpp
						while(result_case_3!=1)
						{
							result_case_3*=-1;
						if(result_case_3&1)
						{
						std:: cout << "Ingrese la matricula" << "\n";
						std::cin >> matricula;
						}
						if(result_case_3&2)
						{
						std:: cout << "Ingrese el nombre" << "\n";
						std::cin >> name;
						}
						if(result_case_3&4)
						{
						std:: cout << "Ingrese el apellido" << "\n";
						std::cin >> last_name;
						}
						if(result_case_3&8)
						{
						std:: cout << "Ingrese el mail" << "\n";
						std::cin >> mail;
						}
						if(result_case_3&16)
						{
						std:: cout << "Ingrese la edad" << "\n";
						std::cin >> age;
						}
						if(result_case_3&32)
						{

						std::cout <<"La matrícula ya está registrada, intente de nuevo."<<std::endl;
						break;
						}
						BiographicalData bio;
						bio.matricula = matricula;
						bio.age = age;
						bio.name = name;
						bio.lastName = last_name;
						bio.mail = mail;
						result_case_3 = face_recognition.enroll(frame, shape, bio);
						std::cout << result_case_3<<std::endl;
						}
						if(result_case_3 == 1){
							std::cout << "La persona fue registrada\n";
						}

					}
	        	flag = false;
				break;

				case 32:
				case 27:
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
