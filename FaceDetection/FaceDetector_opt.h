#include "parameters_FacDet.h"

class FaceDetector_opt
{
private:
  std::vector <cv::CascadeClassifier> classifiers;
  double scaleFact;
  int validNeighbors;
  int minWidth;
  int maxWidth;

public:
  FaceDetector_opt(parameters_FacDet input);

  cv::Rect get_largest_face(std::vector<cv::Rect> faces);

  std::vector<cv::Rect> detect_faces(cv::Mat *img);

  std::vector<cv::Rect> detect_faces(cv::Mat *img, int classifier_index);

  void show_faces(cv::Mat *img, std::vector<cv::Rect> faces);

  void show_faces(cv::Mat *img, std::vector<cv::Rect> detected_faces, std::vector<cv::Rect> real_faces);

  void show_faces(cv::Mat *img, std::vector<cv::Rect> detected_faces, std::vector<cv::Rect> real_faces, cv::Rect largest_face);

  std::vector<cv::Rect> ignore_false_positives(cv::Mat * img, std::vector<cv::Rect> original_det_faces, int tolerance);

  cv::Rect select_additional_face(cv::Mat * img);

  ~ FaceDetector_opt();
};
