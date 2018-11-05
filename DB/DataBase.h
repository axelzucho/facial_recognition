#ifndef DATABASE_H
#define DATABASE_H
//Se incluyen las librer�as necesarias.
#include <bits/stdc++.h>
#include <sys/stat.h>
#include "opencv2/core.hpp"
#include "opencv2/flann/miniflann.hpp"
#include "opencv2/flann/flann_base.hpp"
#include <vector>

using namespace cv::flann;

using std::string;
using cv::Mat;
using std::vector;

class DataBase {
    private:
        Mat queries;    
        Index *flann_index;
        int n;
        string name,lastName,mail,age,biographicalFile,biometricFile,nFile;
        std::fstream biographicalDB,biometricDB,N;

    public:
        DataBase();
        DataBase(string biographicalFile,string biometricFile,string nFile);
        Mat cargarBase(string archivo);
        Mat getMatrix();
        Mat getColumn(int num);
        Mat getRow(int num);
        Mat search(Mat elementoaBuscar,int K);
        void saveUserDataInAFile(std::vector <std::vector <string>> userData);
        void saveUserBiometricDataInAFile(Mat biometric);
        void updateDataBase(int n);
        //~DataBase();
    
};
#endif