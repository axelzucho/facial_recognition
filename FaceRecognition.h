//
// Created by axelzucho on 30/10/18.
//

#ifndef FR_FACERECOGNITION_H
#define FR_FACERECOGNITION_H

//Module 1
#include "FaceDetection/FaceDetector_opt.h"
#include "FaceDetection/parameters_FacDet.h"

//Module 2
#include "FaceAligner/FaceAligner.h"

//Module 3
#include "FaceDescriptorExtractor/FaceDescriptorExtractor.h"

using std::string;

class FaceRecognition {
public:
    FaceRecognition(const parameters_FacDet &parameters, const string &path_to_landmark_model, const unsigned int size,
                    const double left_eye_after, const string &path_to_descriptor_model);


    // Datos es una struct que va a utilizar el equipo 4.
    
    // Caso 1
    // Datos caso1(const Mat* image, dlib::full_object_detection shape)

    // Caso 2
    // bool caso2(const Mat* image, dlib::full_object_detection shape, const string& matricula)

    // Caso 3
    // bool caso3(const Mat* image, dlib::full_object_detection shape, const Datos& datos)

    ~FaceRecognition();

private:
    FaceDetector_opt *face_detector_;
    FaceAligner *face_aligner_;
    FaceDescriptorExtractor *face_descriptor_extactor_;

};


#endif //FR_FACERECOGNITION_H
