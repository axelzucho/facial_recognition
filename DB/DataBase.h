#ifndef DATABASE_H
#define DATABASE_H
//Se incluyen las librerías necesarias.
#include <bits/stdc++.h>
#include <sys/stat.h>
#include "opencv2/core.hpp"
#include "opencv2/flann/miniflann.hpp"
#include "opencv2/flann/flann_base.hpp"


using namespace cv::flann;

using std::string;
using cv::Mat;
using std::vector;

struct BiographicalData{
    string id,matricula,name,lastName,mail,img;
    int age;
};

struct MatriculaId{
    int ID;
    string matricula;
};

class DataBase {
    private:
        Mat queries,ides,descriptores;    
        Index *flann_index;
        int n;
        string matricula,name,lastName,mail,age,biographicalFile,biometricFile,nFile,id_matFile;
        std::fstream biographicalDB,biometricDB,N,Id_Mat;
        std::vector<MatriculaId> Id_MatriculaVector;
       

    public:
        DataBase();
        DataBase(string biographicalFile,string biometricFile,string nFile,string id_matFile);
        //https://stackoverflow.com/questions/12774207/fastest-way-to-check-if-a-file-exist-using-standard-c-c11-c
        inline bool existsFile (const std::string& name);//Return true if the file exists, false if not.
        void cargarBase();
        void cargarId_MatriculaFile();
        Mat getMatrix();
        Mat getColumn(int num);
        Mat getRow(int num);
        Mat search(Mat elementoaBuscar,int K);
        Mat getBiometricByMatricula(string matricula);
        void saveUserDataInAFile(BiographicalData bio);
        void saveUserBiometricDataInAFile(Mat biometric);
        void saveUserImage(Mat &image);
        //bool verify(string matricula,Mat vec);
        void getN();
        void updateDataBase();
    
        //~DataBase();
    
};
#endif
