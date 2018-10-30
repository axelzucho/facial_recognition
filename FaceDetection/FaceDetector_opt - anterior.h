#include "parameters_FacDet.h"

class FaceDetector_opt
{
private:
  vector <CascadeClassifier> classifiers;
  double scaleFact;
  int validNeighbors;
  int minWidth;
  int maxWidth;

public:
  FaceDetector_opt(parameters_FacDet input)
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

  Rect get_largest_face(vector<Rect> faces)
  {
    int largest_size, largest_size_i;
    largest_size_i = 0;
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
  }

  vector<Rect> detect_faces(Mat *img)
  {
    Mat imgGray;//grayscale conversion
    Ptr<CLAHE> clahe;//Contrast Limit Adaptative Histogram Equalization
    vector<Rect> faces;

    clahe = createCLAHE();
    clahe->setClipLimit(4);

    if(!img->empty() && !classifiers.empty())
    {
      //Converting into gray scale and improving contrast
      cvtColor(*img, imgGray, CV_BGR2GRAY);
      clahe->apply(imgGray, imgGray);

      //Detecting faces
      classifiers[0].detectMultiScale(imgGray, faces, scaleFact, validNeighbors, 0|CV_HAAR_SCALE_IMAGE, Size(minWidth, maxWidth));
    }else{
      cout << "Image or classifier not found" << endl;
    }

    return faces;
  }

  vector<Rect> detect_faces(Mat *img, int classifier_index)
  {
    Mat imgGray;//grayscale conversion
    Ptr<CLAHE> clahe;//Contrast Limit Adaptative Histogram Equalization
    vector<Rect> faces;

    clahe = createCLAHE();
    clahe->setClipLimit(4);

    if(!img->empty() && !classifiers.empty())
    {
      //Converting into gray scale and improving contrast
      cvtColor(*img, imgGray, CV_BGR2GRAY);
      clahe->apply(imgGray, imgGray);

      //Detecting faces
      classifiers[classifier_index].detectMultiScale(imgGray, faces, scaleFact, validNeighbors, 0|CV_HAAR_SCALE_IMAGE, Size(minWidth, maxWidth));
    }else{
      cout << "Image or classifier not found" << endl;
    }

    return faces;
  }

  void show_faces(Mat *img, vector<Rect> faces)
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

      rectangle(*img, pt1, pt2, Scalar(rectColor[0], rectColor[1], rectColor[2]));
    }


    //Display window with image
    namedWindow("Face Detection", WINDOW_AUTOSIZE);//Generating window
    imshow("Face Detection", *img);//Showing image

    return;
  }

  void show_faces(Mat *img, vector<Rect> detected_faces, vector<Rect> real_faces)
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

      rectangle(*img, pt1, pt2, Scalar(rectColor[1], rectColor[0], rectColor[2]));
    }

    //Display window with image
    namedWindow("Face Detection", WINDOW_AUTOSIZE);//Generating window
    imshow("Face Detection", *img);//Showing image

    return;
  }

  void show_faces(Mat *img, vector<Rect> detected_faces, vector<Rect> real_faces, Rect largest_face)
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

      rectangle(*img, pt1, pt2, Scalar(rectColor[1], rectColor[0], rectColor[2]));
    }

    //Draw largest face
    pt1.x = largest_face.x;
    pt1.y = largest_face.y;
    pt2.x = largest_face.x + largest_face.width;
    pt2.y = largest_face.y + largest_face.height;

    rectangle(*img, pt1, pt2, Scalar(rectColor[1], rectColor[2], rectColor[0]));

    //Display window with image
    namedWindow("Face Detection", WINDOW_AUTOSIZE);//Generating window
    imshow("Face Detection", *img);//Showing image

    return;
  }

  vector<Rect> ignore_false_positives(Mat * img, vector<Rect> original_det_faces)
  {
    vector<Rect> result;
    vector<vector<Rect>> additional_detections;
    vector<int> coincidence_counter;

    //Initialize coincidences counter with 0
    coincidence_counter.resize(original_det_faces.size(), 0);

    //Iterate for each detection type
    for(int i=1; i<classifiers.size(); i++)
    {
      //Detect with aditional classifiers
      additional_detections.push_back(detect_faces(img, i));

      //iterate for each face in the current detection type
      for(int j=0; j<additional_detections[i-1].size(); j++)
      {
        //Iterate for each face of the original detection type
        for(int k=0; k<original_det_faces.size(); k++)
        {
          if((additional_detections[i-1][j].x >= (original_det_faces[k].x * 0.80)) && (additional_detections[i-1][j].x <= (original_det_faces[k].x * 1.20)))
          {
            if((additional_detections[i-1][j].y >= (original_det_faces[k].y * 0.80)) && (additional_detections[i-1][j].y <= (original_det_faces[k].y * 1.20)))
            {
              coincidence_counter[k]++;
            }
          }
        }
      }
    }

    for(int i=0; i<original_det_faces.size(); i++)
    {
      if(coincidence_counter[i] >= 1)
      {
        result.push_back(original_det_faces[i]);
      }
    }
    return result;
  }

  ~ FaceDetector_opt()
  {

  }
};
