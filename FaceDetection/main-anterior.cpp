#include "FaceDetector_opt.h"

using namespace std;

int main()
{
  //settings and image/video reading variables
  string imgName;
  Mat img;
  parameters_FacDet settings;
  vector <string> pathsToClassif;
  VideoCapture video;

  //path of classifiers to train algorithm
  pathsToClassif.push_back("C:/Users/quiqu/Downloads/OpenCV_pruebas/imgLoadnDisp/classifiers/haarcascade_frontalface_default.xml");

  //Define settings in struct "settings"
  settings.classifiers_location = pathsToClassif;
  settings.scaleFact = 1.1;
  settings.validNeighbors = 2;
  settings.minWidth = 40;
  settings.maxWidth = 40;

  //Initialize detector
  FaceDetector_opt detector(settings);

  /*video.open(0);

  detector.detect_faces_cam(&video);

  cout << "Detectó caras en video" << endl;
  */

  /*vector<vector<Rect>> prueba = detector.detect_faces_video(&video);

  cout << "Regresó vector" << endl;
  cout << "Tamaño del vector: " << prueba.size() << endl;

  for(int i=0; i<prueba.size(); i++)
  {
    for(int j=0; j<prueba[i].size(); j++)
    {
      cout << "X: " << prueba[i][j].x << "Y: " << prueba[i][j].y << endl;
    }
  }
  */


  //Ask the data in CLI
  cout << "Input image file name (with extension)" << endl;
  cout << "Harold.jpg" << endl;
  //imgName = "Harold.jpg";
  imgName = "personas_negocios.png";

  //Input the image as pointer to Mat
  img = imread(imgName, 1);
  cout << "Se guardó la imagen en el Mat (main)" << endl;


  vector<Rect> faces = detector.detect_faces(&img);
  //detector.show_faces(&img, faces);

  Mat img2 = imread("Harold.jpg", 1);
  vector<Rect> faces2 = detector.detect_faces(&img2);

  Rect largest = detector.get_largest_face(faces);

  detector.show_faces(&img, faces, faces2);
  detector.show_faces(&img, faces, faces2, largest);


  //detector.detect_faces_show(&img);

  //Rect face = detector.detect_single_face(&img);

  //vector<Rect> detected_faces = detector.detect_faces(&img);

  cout << "Se detectaron las caras y se regresó el vector" << endl;

  //detector.detect_faces_show(&img);


  return 0;
};
