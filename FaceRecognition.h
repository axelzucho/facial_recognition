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

    // Caso 1

    // Caso 2

    // Caso 3

    ~FaceRecognition();

private:
    FaceDetector_opt *face_detector_;
    FaceAligner *face_aligner_;
    FaceDescriptorExtractor *face_descriptor_extactor_;

};


#endif //FR_FACERECOGNITION_H
