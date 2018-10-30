#include "FaceDetector_opt.h"
#include <vector>
#include <iostream>

using namespace std;
using namespace cv;

int main()
{
  //Variable declaration
  int option;//for menu
  parameters_FacDet settings;//for detector initialization
  vector <string> pathsToClassif;//

  //Image file
  string fileName;
  Mat img;
  //vectors used to display detected faces
  vector<Rect> detected_faces;
  vector<Rect> real_faces;
  vector<Rect> largest_face;

  largest_face.resize(1);

  //Video fileName
  VideoCapture video;
  vector<vector<Rect>> faces_video;
  vector<Rect> temporal;

  //path of classifiers to train algorithm
  pathsToClassif.push_back("C:/Users/quiqu/Downloads/OpenCV_pruebas/imgLoadnDisp/classifiers/haarcascade_frontalface_default.xml");
  pathsToClassif.push_back("C:/Users/quiqu/Downloads/OpenCV_pruebas/imgLoadnDisp/classifiers/haarcascade_frontalface_alt.xml");
  pathsToClassif.push_back("C:/Users/quiqu/Downloads/OpenCV_pruebas/imgLoadnDisp/classifiers/haarcascade_frontalface_alt2.xml");
  pathsToClassif.push_back("C:/Users/quiqu/Downloads/OpenCV_pruebas/imgLoadnDisp/classifiers/haarcascade_frontalface_alt_tree.xml");

  //Define settings in struct "settings"
  settings.classifiers_location = pathsToClassif;
  settings.scaleFact = 1.2;
  settings.validNeighbors = 1;
  settings.minWidth = 30;
  settings.maxWidth = 30;

  //Intitialize detector with settings
  FaceDetector_opt detector(settings);

  //Ask for data in CLI
  cout << "Bienvenido al demo de detección de rostros" << endl;

  do{
    cout << "Introduzca la opcion deseada:" << endl << endl;

    cout << "(1) Detectar rostro de imagen" << endl;
    cout << "(2) Detectar rostro de video" << endl;
    cout << "(3) Detectar rostro de camara" << endl << endl;
    cout << "(0) Salir" << endl;

    cin >> option;

    switch (option)
    {
      case 1: //Detect from image file
        cout << "Introduzca el nombre de la imagen (con extension)" << endl;
        cin >> fileName;

        //read image
        img = imread(fileName, 1);

        if(!img.empty())
        {
          //detect faces and ignore false positives
          detected_faces = detector.detect_faces(&img);
          real_faces = detector.ignore_false_positives(&img, detected_faces, 2);

          //show faces after detection
          detector.show_faces(&img, detected_faces, real_faces);
          waitKey(0);
          destroyAllWindows();

          //Select additional faces manualy
          do{
            cout << "¿Desea agregar un rostro no reconocido?" << endl;
            cout << "(1) Si" << endl;
            cout << "(2) No" << endl;
            cin >> option;

            //User enters into face selection mode
            if(option == 1)
            {
              real_faces.push_back(detector.select_additional_face(&img));
              destroyAllWindows();
            }
          }while(option != 2);

          destroyAllWindows();

          //detect the largest face
          largest_face[0] = detector.get_largest_face(real_faces);
          detector.show_faces(&img, detected_faces, real_faces, largest_face[0]);

          waitKey(0);
          destroyAllWindows();
        }else{
          cout << "No se pudo leer la imagen" << endl;
        }

        break;
      case 2: //Detect from video file
        cout << "Introduzca el nombre del video (con extension)" << endl;
        cin >> fileName;
        video.open(fileName);

        //If vide isn't open
        if(!video.isOpened())
        {
            cout<<"Error: The file/stream couldn't open"<<endl;
        }
        else
        {
            //obtain frames from video
            while(true)
            {
              Mat frame;
              video >> frame;
              if(frame.empty())
              {
                break;
              }
              else
              {
                //image detection for each frame
                temporal = detector.detect_faces(&frame);
                real_faces = detector.ignore_false_positives(&frame, temporal, 3);
                largest_face[0] = detector.get_largest_face(real_faces);
                detector.show_faces(&frame, temporal, real_faces, largest_face[0]);
                waitKey(1);

              
              }
            }
            destroyAllWindows();
            video.release();
        }
        break;
      case 3: //Detect from camera stream
      cout << "Usando la cámara del dispositivo" << endl;
      video.open(0);

      //If video isn't open
      if(!video.isOpened())
      {
          cout<<"Error: The file/stream couldn't open"<<endl;
      }
      else
      {
        while(true)
        {
          Mat frame;
          video >> frame;
          if(frame.empty())
          {
            break;
          }
          else
          {
            //detect faces for each frame
            temporal = detector.detect_faces(&frame);
            real_faces = detector.ignore_false_positives(&frame, temporal, 2);
            largest_face[0] = detector.get_largest_face(real_faces);
            detector.show_faces(&frame, temporal, real_faces, largest_face[0]);
            waitKey(1);

          }
        }
        destroyAllWindows();
        video.release();
      }
        break;
    }

    cout << endl << endl;
  }while(option != 0);

  return 0;
}
