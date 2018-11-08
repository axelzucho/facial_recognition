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

bool result_case_3(const Mat * image, dlib::full_object_detection shape, const BiographicalData data);

int main()
{
	parameters_FacDet settings = initialize_detection_parameters();
    //Some initialization for the first raw example...
	std::vector<cv::Rect> all_faces;
	std::vector<cv::Rect> real_faces;
  	std::vector<cv::Rect> largest_face;
  	largest_face.resize(1);
	//path of classifiers to train algorithm
  	FaceRecognition face_recognition (settings, "../FaceAligner/shape_predictor_5_face_landmarks.dat", 150, 0.3, "../FaceDescriptorExtractor/dlib_face_recognition_resnet_model_v1.dat");
  	FaceAligner face_transformer("../FaceAligner/shape_predictor_5_face_landmarks.dat", 150, 0.3);
	//Create database object
  	cv::Mat template_image;
  	dlib::full_object_detection shape;

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
			std::cout << "1";
			cv::destroyAllWindows();
        	video.release();
        	std:: cout << "Ingrese la matricula" << "\n";
				// Aqui se implementa caso 1
				{
					string matricula;
					std::cin >> matricula;
					std::pair<int, BiographicalData> result_case_1;
					result_case_1 = face_recognition.caso1(&frame, shape, matricula);
				}
        	flag = false;
			break;

			case '2':
			std::cout << "2";
			cv::destroyAllWindows();
        	video.release();
        	// Aqui se implementa caso 2
				{
					std::pair<int, BiographicalData> result_case_2;
					result_case_2 = face_recognition.caso2(&frame, shape);
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
					int age;
					std:: cout << "Ingrese la matricula" << "\n";
					std::cin >> matricula;
					std:: cout << "Ingrese el nombre" << "\n";
					std::cin >> name;
					std:: cout << "Ingrese el apellido" << "\n";
					std::cin >> last_name;
					std:: cout << "Ingrese el mail" << "\n";
					std::cin >> mail;
					std:: cout << "Ingrese la edad" << "\n";
					std::cin >> age;
					BiographicalData bio;
					bio.matricula = matricula;
					bio.age = age;
					bio.name = name;
					bio.lastName = last_name;
					bio.mail = mail;
					int errorCase3 = face_recognition.database_->ValidateData(&bio);
					if (errorCase3 != 0){
						//Aquí se imprime el error, debe haber un loop si los datos no están correctos.
						std::cout<<errorCase3<<std::endl;
					}else{
						bool result_case_3;
						result_case_3 = face_recognition.enroll(frame, shape, bio);
					}
					
				}
        	flag = false;
			break;
			
			default:
				break;
		}
		
	}
		
	return 0;
}


