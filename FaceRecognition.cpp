//
// Created by axelzucho on 30/10/18.
//

#include "FaceRecognition.h"

FaceRecognition::FaceRecognition(const parameters_FacDet &parameters, const string &path_to_landmark_model,
                                 const unsigned int size, const double left_eye_after,
                                 const string &path_to_descriptor_model, float threshold) {
    face_detector_ = new FaceDetector_opt(parameters);
    face_aligner_ = new FaceAligner(path_to_landmark_model, size, left_eye_after);
    face_descriptor_extactor_ = new FaceDescriptorExtractor(path_to_descriptor_model);
    database_ = new DataBase();
    threshold_ = threshold;
}

FaceRecognition::~FaceRecognition() {
    delete(face_detector_);
    delete(face_aligner_);
    delete(face_descriptor_extactor_);
    delete(database_);
}

std::pair<int, BiographicalData> FaceRecognition::caso1(const Mat *image, dlib::full_object_detection shape,
                                                         const string &matricula) {
    cv::Mat template_image;//aquí se guarda la imagen ya alineada
    face_aligner_->Align(shape, *image, template_image);//se alínea la imagen
    //mostramos la imagen
    cv::imshow("Face", template_image);
    cv::waitKey(0);
    //obtenemos los descriptores de la persona que solicita acceso
    Mat face = face_descriptor_extactor_->obtenerDescriptorVectorial(template_image);
    //obtenemos los descriptores guardados de la matrícula en la base de datos
    Mat face_db = database_->getBiometricByMatricula(matricula);
    //guardamos la distancia euclidana entre los dos Mats que incluyen los descriptores
    float resultado_inspec = face_descriptor_extactor_->compararDescriptores(face, face_db);
    //Comparamos el resultado con el threshold para dar acceso o no
    if(resultado_inspec == -2)//error detectado
    {//en caso de que la matrícula no exista
        return {-2, BiographicalData()};
    }
    else if(resultado_inspec < 0)
    {//en caso de cualquier error
        return{-1, BiographicalData()};
    }
    else if(resultado_inspec < threshold_)
    {//en caso de ser la misma persona
        return{1, BiographicalData()};
    }
    else if(resultado_inspec > threshold_)
    {//en caso de que no sea la misma persona guardada en la base de datos
        return{0, BiographicalData()};
    }
    return {0, BiographicalData()};
}

std::pair<int, BiographicalData> FaceRecognition::caso2(const Mat *image, dlib::full_object_detection shape) {

  Mat template_image;
  BiographicalData output_biographical_data;
  std::pair<Mat, Mat> output_mat;
  float valor, distancia;

  //Alinear la imagen
  face_aligner_->Align(shape, *image, template_image);

  //Obtener los rasgos del rostro
  template_image =  face_descriptor_extactor_->obtenerDescriptorVectorial(template_image);

  //Comparar con la base de datos
  output_mat = database_->search(template_image, 1);

  valor = output_mat.first.at<float>(0,0);
  distancia = output_mat.second.at<float>(0,0);

  if(distancia < threshold_)
  {
    output_biographical_data = database_->getUserInfoByID(int(valor));
    return {1, output_biographical_data};
  }else{
    return {0, BiographicalData()};
  }
}

int FaceRecognition::enroll(const Mat &image, dlib::full_object_detection shape, const BiographicalData datos) {
   database_->getN();
   database_->saveUserDataInAFile(datos);
   Mat templ;
   face_aligner_->Align(shape,image,templ);
   database_->saveUserImage(templ);
   Mat res;
   res = face_descriptor_extactor_->obtenerDescriptorVectorial(templ);
   database_->saveUserBiometricDataInAFile(res);
   database_->updateDataBase();
   return 1;
}
