//
// Created by axelzucho on 30/10/18.
//

#include "FaceRecognition.h"
#include "opencv2/opencv.hpp"

#include <iostream>
#include <vector>

int main()
{
	//Caso1 - Reconocer una persona en camara

	parameters_FacDet settings;//for detector initialization
	std::vector <string> pathsToClassif;
	std::vector<cv::Rect> temporal;
	std::vector<cv::Rect> real_faces;
  	std::vector<cv::Rect> largest_face;
  	largest_face.resize(1);
	//path of classifiers to train algorithm
  	pathsToClassif.push_back("/home/rodrigo/Documents/Facial Recognition/Final/facial_recognition/FaceDetection/classifiers/haarcascade_frontalface_default.xml");
  	pathsToClassif.push_back("/home/rodrigo/Documents/Facial Recognition/Final/facial_recognition/FaceDetection/classifiers/haarcascade_frontalface_alt.xml");
  	pathsToClassif.push_back("/home/rodrigo/Documents/Facial Recognition/Final/facial_recognition/FaceDetection/classifiers/haarcascade_frontalface_alt2.xml");
  	pathsToClassif.push_back("/home/rodrigo/Documents/Facial Recognition/Final/facial_recognition/FaceDetection/classifiers/haarcascade_frontalface_alt_tree.xml");
  	settings.classifiers_location = pathsToClassif;
  	settings.scaleFact = 1.2;
  	settings.validNeighbors = 1;
  	settings.minWidth = 30;
  	settings.maxWidth = 30;
  	FaceRecognition face_recognition (settings, "/home/rodrigo/Documents/Facial Recognition/Final/facial_recognition/FaceAligner/shape_predictor_5_face_landmarks.dat", 500, 0.3, "/home/rodrigo/Documents/Facial Recognition/Final/facial_recognition/FaceDescriptorExtractor/dlib_face_recognition_resnet_model_v1.dat");
  	FaceDetector_opt detector (settings);
	
  	FaceAligner face_transformer("/home/rodrigo/Documents/Facial Recognition/Final/facial_recognition/FaceAligner/shape_predictor_5_face_landmarks.dat", 500, 0.3);
  	cv::Mat template_image;
  	dlib::full_object_detection shape;

	cv::VideoCapture video(0);
	video.open(0);
	while (!video.isOpened()) {
	    std::cout << "Error al intentar abrir la camara" << std::endl;
	}
	int opcion;
	std::cout << "Ingrese el número de la operación deseada:\n1. Reconocer una persona en camara\n2. Verificar si la persona ubicada coincide con la matricula ingresada\n3. Enrolar una persona nueva" << std::endl;
	bool flag = true;
	while (flag)
	{
		cv::Mat frame;
		video>> frame;
		temporal = detector.detect_faces(&frame);
		real_faces = detector.ignore_false_positives(&frame, temporal, 2);
		largest_face[0] = detector.get_largest_face(real_faces);
		//detector.show_faces(&frame, temporal, real_faces, largest_face[0], shape);
		detector.show_faces(&frame, temporal, real_faces, largest_face[0]);

		
		//cv::imshow("Camara", frame);
		//cv::waitKey(1);
		char Key_pressed=cv::waitKey(1);
		switch(Key_pressed)
		{
			case '1':
			std::cout << "1";
			//face_transformer.Detect(frame, largest_face[0], shape);
			cv::destroyAllWindows();
        	video.release();
        	flag = false;
			/*temporal = detector.detect_faces(&frame);
			real_faces = detector.ignore_false_positives(&frame, temporal, 2);
			largest_face[0] = detector.get_largest_face(real_faces);
			detector.show_faces(&frame, temporal, real_faces, largest_face[0]);
			face_transformer.Detect(frame, largest_face[0], shape);*/


			cv::waitKey(0);
			
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
