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


int main()
{
	parameters_FacDet settings = initialize_detection_parameters();
    //Some initialization for the first raw example...
	std::vector<cv::Rect> all_faces;
	std::vector<cv::Rect> real_faces;
  	std::vector<cv::Rect> largest_face;
  	largest_face.resize(1);
	//path of classifiers to train algorithm
  	FaceRecognition face_recognition (settings, "../FaceAligner/shape_predictor_5_face_landmarks.dat", 500, 0.3, "../FaceDescriptorExtractor/dlib_face_recognition_resnet_model_v1.dat");

  	FaceAligner face_transformer("../FaceAligner/shape_predictor_5_face_landmarks.dat", 500, 0.3);
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
		face_recognition.face_detector_->show_faces(&frame, all_faces, real_faces, largest_face[0]);
		char key_pressed = cv::waitKey(1);
		switch(key_pressed)
		{
			case '1':
			std::cout << "1";
			//face_recognition.face_aligner_->Detect(frame, largest_face[0], shape);
			cv::destroyAllWindows();
        	video.release();
        	flag = false;
			break;

			case '2':
			std::cout << "2";
			std:: cout << "Ingrese la matricula" << std::endl;
			cv::destroyAllWindows();
        	video.release();
        	flag = false;
			break;

			case '3':
			std::cout << "1";
			cv::destroyAllWindows();
        	video.release();
        	flag = false;
			break;
			
			default:
				break;
		}
		
	}
		
	return 0;
}
