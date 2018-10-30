#ifndef PARAMETERS_FACEDET_H
#define PARAMETERS_FACEDET_H

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/tracking.hpp>
#include <string.h>
#include <vector>

typedef struct parameters_FacDet{
  std::vector <std::string> classifiers_location;

  //detectMultiScale
  double scaleFact;
  int validNeighbors;
  int minWidth;
  int maxWidth;
}parameters_FacDet;

#endif