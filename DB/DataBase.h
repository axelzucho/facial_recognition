#ifndef DATABASE_H
#define DATABASE_H
//Se incluyen las librerías necesarias.
#include <sys/stat.h>
#include <string>
#include <iostream>
#include <fstream>
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
        std::vector<BiographicalData> biograData;
       

    public:
        DataBase();
        DataBase(string biographicalFile,string biometricFile,string nFile,string id_matFile);
        //https://stackoverflow.com/questions/12774207/fastest-way-to-check-if-a-file-exist-using-standard-c-c11-c
        inline bool existsFile (const std::string& name);//Return true if the file exists, false if not.
        void load_BiometricFile();
        void load_ImgFolder();
        void load_N_File();
        void load_Id_MatriculaFile();
        void load_BiographicalFile();
        Mat getMatrix();
        Mat getColumn(int num);
        Mat getRow(int num);
        Mat search(Mat elementoaBuscar,int K);
        Mat getBiometricByMatricula(string matricula);
        void saveUserDataInAFile(BiographicalData bio);
        void saveUserBiometricDataInAFile(Mat biometric);
        void saveUserImage(Mat &image);
        void getN();
        void updateDataBase();
        bool ValidName(std::string word);
        bool SimpleValidateMail(std::string mail);
        int ValidateData(BiographicalData* Data);
        BiographicalData getUserInfoByID(int ID);
        BiographicalData String_To_Structure(std::string Data_As_String);
        std::vector<std::string> indexData(std::string dataLine);
    
        //~DataBase();
    
};
#endif
