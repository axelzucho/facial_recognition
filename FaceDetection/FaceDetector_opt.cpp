#include "FaceDetector_opt.h"

using namespace cv;
using namespace std;

FaceDetector_opt::FaceDetector_opt(parameters_FacDet input)
{
  //pass data from struct
  scaleFact = input.scaleFact;
  validNeighbors = input.validNeighbors;
  minWidth = input.minWidth;
  maxWidth = input.maxWidth;

  //load classifiers
  CascadeClassifier tmp;

  for(int i=0; i<input.classifiers_location.size(); i++)
  {
    tmp.load(input.classifiers_location[i]);
    classifiers.push_back(tmp);
  }
}

Rect FaceDetector_opt::get_largest_face(vector<Rect> faces)
{
  Rect empty;
  int largest_size, largest_size_i;
  largest_size_i = 0;

  if(!faces.empty())
  {
    largest_size = faces[0].width;

    for(int i=1; i<faces.size(); i++)
    {
      if(faces[i].width > largest_size)
      {
        largest_size = faces[i].width;
        largest_size_i = i;
      }
    }

    return faces[largest_size_i];
  }else{
    return empty;
  }
}

vector<Rect> FaceDetector_opt::detect_faces(Mat *img)
{
  Mat imgProc;//Mat to store the image after being procesed
  Ptr<CLAHE> clahe;//Contrast Limit Adaptative Histogram Equalization
  vector<Rect> faces;//stores the faces detected

  //Variables for resize
  float opt_width = 300;
  float frame_ratio;
  float opt_height;
  float resize_per;

  clahe = createCLAHE();
  clahe->setClipLimit(4);

  imgProc = *img;

  //Get percentage of growth
  resize_per = (opt_width/imgProc.size().width);

  //Resize to optimize detection
  frame_ratio = (float)imgProc.size().height / (float)imgProc.size().width;
  opt_height = opt_width*frame_ratio;
  resize(*img, imgProc, Size((int)opt_width, (int)opt_height));

  //Converting into gray scale and improving contrast
  cvtColor(imgProc, imgProc, CV_BGR2GRAY);
  clahe->apply(imgProc, imgProc);

  //Detecting faces
  classifiers[0].detectMultiScale(imgProc, faces, scaleFact, validNeighbors, 0|CV_HAAR_SCALE_IMAGE, Size(minWidth, maxWidth));

  //Resize back Rects obtained in detection
  for(int i=0; i<faces.size(); i++)
  {
    faces[i].x = (int)((float)faces[i].x / resize_per);
    faces[i].y = (int)((float)faces[i].y / resize_per);
    faces[i].width = (int)((float)faces[i].width / resize_per);
    faces[i].height = (int)((float)faces[i].height / resize_per);
  }
  return faces;
}

vector<Rect> FaceDetector_opt::detect_faces(Mat *img, int classifier_index)
{
  Mat imgProc;//Mat to store the image after being procesed
  Ptr<CLAHE> clahe;//Contrast Limit Adaptative Histogram Equalization
  vector<Rect> faces;//stores the faces detected

  //Variables for resize
  float opt_width = 300;
  float frame_ratio;
  float opt_height;
  float resize_per;

  clahe = createCLAHE();
  clahe->setClipLimit(4);

  imgProc = *img;

  //Get percentage of growth
  resize_per = (opt_width/imgProc.size().width);

  //Resize to optimize detection
  frame_ratio = (float)imgProc.size().height / (float)imgProc.size().width;
  opt_height = opt_width*frame_ratio;
  resize(*img, imgProc, Size((int)opt_width, (int)opt_height));

  //Converting into gray scale and improving contrast
  cvtColor(imgProc, imgProc, CV_BGR2GRAY);
  clahe->apply(imgProc, imgProc);

  //Detecting faces
  classifiers[classifier_index].detectMultiScale(imgProc, faces, scaleFact, validNeighbors, 0|CV_HAAR_SCALE_IMAGE, Size(minWidth, maxWidth));

  //Resize back Rects obtained in detection
  for(int i=0; i<faces.size(); i++)
  {
    faces[i].x = (int)((float)faces[i].x / resize_per);
    faces[i].y = (int)((float)faces[i].y / resize_per);
    faces[i].width = (int)((float)faces[i].width / resize_per);
    faces[i].height = (int)((float)faces[i].height / resize_per);
  }
  return faces;
}

void FaceDetector_opt::show_faces(Mat *img, vector<Rect> faces)
{
  Point pt1, pt2;
  vector <int> rectColor;

  rectColor.push_back(255);
  rectColor.push_back(0);
  rectColor.push_back(0);

  for(size_t i=0; i<faces.size(); i++)
  {
    //Drawing rectangle on image
    pt1.x = faces[i].x;
    pt1.y = faces[i].y;
    pt2.x = faces[i].x + faces[i].width;
    pt2.y = faces[i].y + faces[i].height;

    //blue
    rectangle(*img, pt1, pt2, Scalar(rectColor[0], rectColor[1], rectColor[2]));
  }


  //Display window with image
  namedWindow("Face Detection", WINDOW_AUTOSIZE);//Generating window
  imshow("Face Detection", *img);//Showing image

  return;
}

void FaceDetector_opt::show_faces(Mat *img, vector<Rect> detected_faces, vector<Rect> real_faces)
{
  Point pt1, pt2;
  vector <int> rectColor;

  rectColor.push_back(255);
  rectColor.push_back(0);
  rectColor.push_back(0);

  //Draw detected faces
  for(size_t i=0; i<detected_faces.size(); i++)
  {
    //Drawing rectangle on image
    pt1.x = detected_faces[i].x;
    pt1.y = detected_faces[i].y;
    pt2.x = detected_faces[i].x + detected_faces[i].width;
    pt2.y = detected_faces[i].y + detected_faces[i].height;

    //blue
    rectangle(*img, pt1, pt2, Scalar(rectColor[0], rectColor[1], rectColor[2]));
  }

  //Draw real faces (after discarding false positives)
  for(size_t i=0; i<real_faces.size(); i++)
  {
    //Drawing rectangle on image
    pt1.x = real_faces[i].x;
    pt1.y = real_faces[i].y;
    pt2.x = real_faces[i].x + real_faces[i].width;
    pt2.y = real_faces[i].y + real_faces[i].height;

    //green
    rectangle(*img, pt1, pt2, Scalar(rectColor[1], rectColor[0], rectColor[2]));
  }

  //Display window with image
  namedWindow("Face Detection", WINDOW_AUTOSIZE);//Generating window
  imshow("Face Detection", *img);//Showing image

  return;
}

void FaceDetector_opt::show_faces(Mat *img, vector<Rect> detected_faces, vector<Rect> real_faces, Rect largest_face)
{
  Point pt1, pt2;
  vector <int> rectColor;

  rectColor.push_back(255);
  rectColor.push_back(0);
  rectColor.push_back(0);

  //Draw detected faces
  for(size_t i=0; i<detected_faces.size(); i++)
  {
    //Drawing rectangle on image
    pt1.x = detected_faces[i].x;
    pt1.y = detected_faces[i].y;
    pt2.x = detected_faces[i].x + detected_faces[i].width;
    pt2.y = detected_faces[i].y + detected_faces[i].height;

    //blue
    rectangle(*img, pt1, pt2, Scalar(rectColor[0], rectColor[1], rectColor[2]));
  }

  //Draw real faces (after discarding false positives)
  for(size_t i=0; i<real_faces.size(); i++)
  {
    //Drawing rectangle on image
    pt1.x = real_faces[i].x;
    pt1.y = real_faces[i].y;
    pt2.x = real_faces[i].x + real_faces[i].width;
    pt2.y = real_faces[i].y + real_faces[i].height;

    //green
    rectangle(*img, pt1, pt2, Scalar(rectColor[1], rectColor[0], rectColor[2]));
  }

  //Draw largest face
  pt1.x = largest_face.x;
  pt1.y = largest_face.y;
  pt2.x = largest_face.x + largest_face.width;
  pt2.y = largest_face.y + largest_face.height;

  //red
  rectangle(*img, pt1, pt2, Scalar(rectColor[1], rectColor[2], rectColor[0]));

  //Display window with image
  namedWindow("Face Detection", WINDOW_AUTOSIZE);//Generating window
  imshow("Face Detection", *img);//Showing image

  return;
}

vector<Rect> FaceDetector_opt::ignore_false_positives(Mat * img, vector<Rect> original_det_faces, int tolerance)
{
  vector<Rect> result;
  vector<vector<Rect>> additional_detections;
  vector<int> coincidence_counter;
  Rect tmp;

  Mat local_region;
  Mat imgReceived = *img;

  //Initialize coincidences counter with 0
  coincidence_counter.resize(original_det_faces.size(), 0);

  //Iterates over each face originaly detected
  for(int i=0; i<original_det_faces.size(); i++)
  {
    if(!original_det_faces[i].empty())
    {
      local_region = imgReceived(original_det_faces[i]);

      //Iterate over each detection type
      for(int j=1; j<classifiers.size(); j++)
      {
        additional_detections.push_back(detect_faces(&local_region, i));
        if(additional_detections[i].size() >= 1)
        {
          coincidence_counter[i]++;
        }
      }
    }
  }

  for(int i=0; i<original_det_faces.size(); i++)
  {
    if(coincidence_counter[i] >= tolerance)
    {
      result.push_back(original_det_faces[i]);
    }
  }

  return result;
}

Rect FaceDetector_opt::select_additional_face(Mat * img)
{
  Rect output = selectROI(*img);

  if(output.size().width > output.size().height)
  {
    output.height = output.size().width;
  }else{
    output.width = output.size().height;
  }

  return output;
}

FaceDetector_opt::~FaceDetector_opt()
{

}
