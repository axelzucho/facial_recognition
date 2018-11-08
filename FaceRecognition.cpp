//
// Created by axelzucho on 30/10/18.
//

#include "FaceRecognition.h"

FaceRecognition::FaceRecognition(const parameters_FacDet &parameters, const string &path_to_landmark_model,
                                 const unsigned int size, const double left_eye_after,
                                 const string &path_to_descriptor_model) {
    face_detector_ = new FaceDetector_opt(parameters);
    face_aligner_ = new FaceAligner(path_to_landmark_model, size, left_eye_after);
    face_descriptor_extactor_ = new FaceDescriptorExtractor(path_to_descriptor_model);
    database_ = new DataBase();
}

FaceRecognition::~FaceRecognition() {
    delete(face_detector_);
    delete(face_aligner_);
    delete(face_descriptor_extactor_);
    delete(database_);
}

std::pair<int, BiographicalData> FaceRecognition::caso1(const Mat *image, dlib::full_object_detection shape,
                                                         const string &matricula) {
    return {0, BiographicalData()};
}

std::pair<int, BiographicalData> FaceRecognition::caso2(const Mat *image, dlib::full_object_detection shape) {
    return {0, BiographicalData()};
}

bool FaceRecognition::caso3(const Mat &image, dlib::full_object_detection shape, const BiographicalData datos) {
   database_->getN(); 
   database_->saveUserDataInAFile(datos);
   database_->updateDataBase();
   Mat templ;
   face_aligner_->Align(shape,image,templ);
   database_->saveUserImage(templ);
   Mat res;
   //res = face_descriptor_extactor_->obtenerDescriptorVectorial(templ);
   //database_->saveUserBiometricDataInAFile(res);
    return false;
}