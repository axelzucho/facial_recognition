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

//Module 4
#include "DB/DataBase.h"


using std::string;

class FaceRecognition {
public:
    FaceRecognition(const parameters_FacDet &parameters, const string &path_to_landmark_model, const unsigned int size,
                    const double left_eye_after, const string &path_to_descriptor_model, float threshold, int neighbor_quantity);


    // Datos es una struct que va a utilizar el equipo 4.

    // Caso 1
    std::pair<int, BiographicalData> caso1(const Mat* image, dlib::full_object_detection shape, const string& matricula);

    // Caso 2
    std::pair<int, std::vector<std::pair<BiographicalData, float>>> caso2(const Mat* image, dlib::full_object_detection shape);

    // Caso 3
    int enroll(const Mat &image, dlib::full_object_detection shape, const BiographicalData &datos);

    ~FaceRecognition();

//private:
    FaceDetector_opt *face_detector_;
    FaceAligner *face_aligner_;
    FaceDescriptorExtractor *face_descriptor_extactor_;
    DataBase *database_;
    float threshold_;
    int neighbor_quantity_;

};


#endif //FR_FACERECOGNITION_H
