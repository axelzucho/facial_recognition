//
// Created by axelzucho on 30/10/18.
//

#include "FaceRecognition.h"
#define ALIGN_ERR -1
#define EXTRACTOR_ERR -2
#define DB_SEARCH_ERR -4
#define DB_INFO_ERR -8
#define DUPLMAT -32

FaceRecognition::FaceRecognition(const parameters_FacDet &parameters, const string &path_to_landmark_model,
                                 const unsigned int size, const double left_eye_after,
                                 const string &path_to_descriptor_model, float threshold, int neighbor_quantity) {
    face_detector_ = new FaceDetector_opt(parameters);
    face_aligner_ = new FaceAligner(path_to_landmark_model, size, left_eye_after);
    face_descriptor_extactor_ = new FaceDescriptorExtractor(path_to_descriptor_model);
    database_ = new DataBase();
    threshold_ = threshold;
    neighbor_quantity_ = neighbor_quantity;
}

FaceRecognition::~FaceRecognition() {
    delete(face_detector_);
    delete(face_aligner_);
    delete(face_descriptor_extactor_);
    delete(database_);
}

std::pair<int, BiographicalData> FaceRecognition::caso1(const Mat *image, dlib::full_object_detection shape,
                                                         const string &matricula) {

    BiographicalData datos;
    cv::Mat template_image;//aquí se guarda la imagen ya alineada
    face_aligner_->Align(shape, *image, template_image);//se alínea la imagen

    //obtenemos los descriptores de la persona que solicita acceso
    Mat face = face_descriptor_extactor_->obtenerDescriptorVectorial(template_image);
    //obtenemos los descriptores guardados de la matrícula en la base de datos
    Mat face_db = database_->getBiometricByMatricula(matricula);
    //guardamos la distancia euclidana entre los dos Mats que incluyen los descriptores
    float resultado_inspec = face_descriptor_extactor_->compararDescriptores(face, face_db);
    //Comparamos el resultado con el threshold para dar acceso o no
    if(resultado_inspec == EXTRACTOR_ERR)//error detectado
    {//en caso de que la matrícula no exista
        return {EXTRACTOR_ERR, BiographicalData()};
    }
    else if(resultado_inspec < 0)
    {//en caso de cualquier error
        return{ALIGN_ERR, BiographicalData()};
    }
    else if(resultado_inspec < threshold_)
    {//en caso de ser la misma persona
        datos = database_->getUserInfoByMatricula(matricula);
        return{1, datos};
    }
    else if(resultado_inspec > threshold_)
    {//en caso de que no sea la misma persona guardada en la base de datos
        return{0, BiographicalData()};
    }
    return {0, BiographicalData()};
}

std::pair <int, std::vector<std::pair<BiographicalData, float>>> FaceRecognition::caso2(const Mat *image, dlib::full_object_detection shape) {

  bool found = false;
  Mat template_image;
  std::vector<std::pair<BiographicalData, float>> output_biographical_data;
  std::pair<Mat, Mat> output_mat;
  int index;
  float distance;
  std::pair <BiographicalData, float> tmp;
  int accum_error = 0;

  //debugging<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
  //std::cout << "neighbor_quantity=" << neighbor_quantity_ << std::endl;
  //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

  //Alinear la imagen
  face_aligner_->Align(shape, *image, template_image);

  //Obtener los rasgos del rostro
  template_image =  face_descriptor_extactor_->obtenerDescriptorVectorial(template_image);

  //Comparar con la base de datos
  output_mat = database_->search(template_image, neighbor_quantity_);

  //Obtener la información de los "neighbor_quantity_" rostros más cercanos que cumplan con el threshold_
  for(int i=0; i<neighbor_quantity_; i++)
  {
    index = output_mat.first.at<int>(i,0);
    distance = output_mat.second.at<float>(i,0);

    //debugging<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    //std::cout << "Valor: " << index << std::endl;
    //std::cout << "Distancia: " << distance << std::endl;
    //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    tmp.first = database_->getUserInfoByID(index);
    tmp.second = distance;
    output_biographical_data.push_back(tmp);

    //debugging<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    //std::cout << "Se obtuvo información, index=" << output_biographical_data.back().first.id << std::endl;
    //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    if(distance < threshold_)
    {
      found = true;
    }
  }

  //debugging<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
  //std::cout << "Salió del for" << std::endl;
  //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


  return {found, output_biographical_data};
}

int FaceRecognition::enroll(const Mat &image, dlib::full_object_detection shape, const BiographicalData &datos) {
  int result_enroll= database_->ValidateData(&datos);
  bool checkMat = database_->DuplicatedMatricula(datos.matricula);
  if(checkMat == true){
    std::cout<< "Matrícula Duplicada"<<std::endl;
    result_enroll += DUPLMAT;
  }

  if(result_enroll >= 0)
  {
    std::vector<cv::Point2f> points;

    dlib::point temPoint;
    for (size_t i = 0; i < shape.num_parts(); i++)
    {
      temPoint = shape.part(i);

      //std::cout<<"x: "<<temPoint.x()<<"y: "<<temPoint.y()<<std::endl;
      points.push_back(cv::Point2f(temPoint.x(), temPoint.y()));

    }

    CV_Assert(database_->getN());
    CV_Assert(database_->saveUserDataInAFile(datos,points));
    CV_Assert(database_->saveId_Matricula(datos));
    Mat i = image.clone();
    Mat templ;
    database_->saveUserImage(i);
    face_aligner_->Align(shape,image,templ);
    Mat res;
    res = face_descriptor_extactor_->obtenerDescriptorVectorial(templ);
    CV_Assert(database_->saveUserBiometricDataInAFile(res));
    CV_Assert(database_->updateDataBase());
    result_enroll=1;//Success
  }
   return result_enroll;
}
