#include "DataBase.h"

#include <vector>

using std::vector;
using std::cout;
using std::string;
using std::fstream;
using std::ios;

DataBase::DataBase(string biographicalFile,string biometricFile,string nFile){
    this->biographicalFile=biographicalFile;
    this->biometricFile=biometricFile;
    this->nFile=nFile;
    queries =  cargarBase(biometricFile);
    flann_index = new Index(queries, cv::flann::KDTreeIndexParams());
     N.open(nFile,ios::in);
        if(N.is_open()){
            N>>n;
            N.close();
        }else cout<<"Unable to open N.txt\n";
}

DataBase::DataBase(){
    string archivo = "biometrics.txt";
    string narch = "N.txt";
     N.open(narch,ios::out);
        if(N.is_open()){
            N<<"0\n";
            cout<<"***Fichero N.txt creado con éxito***\n";
            n=0;
            N.close();
        }else cout<< "Unable creating N.txt\n";
        
        if(mkdir("Fotos", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == 0){
            cout<<"Directory successfully created"<<'\n';
            
        }else cout<<"Error creating directory or the directory already exits"<<'\n';
    
   /* queries =  cargarBase(archivo);
    flann_index = new Index(queries, cv::flann::KDTreeIndexParams());   */
}

Mat DataBase::cargarBase(string archivo){
   int rows = 0;
   Mat res;
    std::string line;
    fstream arch;
    arch.open(archivo);
    while (getline(arch, line)) {  
        line+=',';
        std::istringstream stream(line);
        char sep; //comma!
        float x;
        while (stream >> x && stream >> sep) {
            res.push_back(x);
        }
        rows ++;
    } 

    // reshape to 2d:
    res = res.reshape(1,rows);
    return res;
}



 Mat DataBase::getMatrix(){
    return queries;
 }

 Mat DataBase::getColumn(int num){
    return queries.col(num);
 }

 Mat DataBase::getRow(int num){
    return queries.row(num);
 }
 

 Mat DataBase::search(Mat elementoaBuscar,int K){
    clock_t tStart = clock();
    Mat indices,dists;
    flann_index->knnSearch(elementoaBuscar,indices,dists,K);
    printf("Time taken: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
    return indices; 
 }

 void DataBase::saveUserDataInAFile(vector <vector <string>> userData){
    int id=n;
    biographicalDB.open(biographicalFile,ios::out | ios::app);
    
    if(biographicalDB.is_open()){
        for(int i=0; i<userData.size(); i++){
            biographicalDB<<id;
            for(int j=0; j<userData[i].size(); j++){
                biographicalDB<<","<<userData[i][j];
            }
            biographicalDB<<"\n";
            id++;
        }
        biographicalDB.close();
    }else cout<<"Unable to open file\n";
    updateDataBase(id);
}
void DataBase::updateDataBase(int n){
    N.open(nFile);
    
    if(N.is_open()){
        N<<n;
        N.close();
    }else cout<<"Error updating N.txt file\n";
}
void DataBase::saveUserBiometricDataInAFile(Mat biometric){
    
    biometricDB.open(biometricFile,ios::out | ios::app);
    cout<<biometric.cols<<std::endl;
    //biometricDB<<endl;
    cv::FileStorage file(biometricFile, cv::FileStorage::WRITE);
//cv::Mat someMatrixOfAnyType;

// Write to file!
    //file << biometric;
    //file << biometric;
    for(int i=0; i<biometric.rows; i++){
             biometricDB<<biometric;
         
    }
    //biometricDB<<"\n";
    n++;
}

 