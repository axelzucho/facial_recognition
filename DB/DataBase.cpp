#include "DataBase.h"
#include "opencv2/opencv.hpp"

DataBase::DataBase(){
    this->biographicalFile= "Data.txt";
    this->biometricFile = "biometrics.txt";
    this->nFile = "N.txt";

    if(!existsFile(nFile)){
     N.open(nFile,std::ios::out);
        if(N.is_open()){
            N<<"0\n";
            std::cout<<"***Fichero N.txt creado con éxito***\n";
            N.close();
        }else std::cout<< "Unable creating N.txt\n";
    }
        
        if(mkdir("Fotos", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == 0){
            std::cout<<"Directory successfully created"<<'\n';
            
        }else std::cout<<"Error creating directory or the directory already exits"<<'\n';
    
   /* queries =  cargarBase(archivo);
    flann_index = new Index(queries, cv::flann::KDTreeIndexParams());   */
}

DataBase::DataBase(string biographicalFile,string biometricFile,string nFile,string id_matFile){
    this->biographicalFile=biographicalFile;
    this->biometricFile=biometricFile;
    this->nFile=nFile;
    this->id_matFile=id_matFile;
    cargarBase();
    cargarId_MatriculaFile();
    //cout<<descriptores<<endl;
    flann_index = new Index(descriptores, cv::flann::KDTreeIndexParams());
    std::cout<<"listo"<<std::endl;
}

inline bool DataBase::existsFile (const std::string& name) {
    std::ifstream f(name.c_str());
    return f.good();
    
}

void DataBase::cargarBase(){
   int rows = 0;
   Mat res;
    std::string line;
    biometricDB.open(biometricFile);
    if(biometricDB.is_open()){
        while (getline(biometricDB, line)) {
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
        
        cv::Rect ids_rect(0,0,1,res.rows);
        cv::Rect descr_rect(1, 0, res.cols-1,res.rows);
        Mat ids(res, ids_rect);
        ides = ids.clone();
        Mat descr(res, descr_rect);
        //cout<<descr<<endl;
        biometricDB.close();
        descriptores = descr.clone();
        
        
    }else std::cout<<"Unable to open: "<<biometricFile<<'\n';

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

void DataBase::cargarId_MatriculaFile(){
    MatriculaId m;

    Id_Mat.open(id_matFile,std::ios::in);
    
    if(Id_Mat.is_open()){
       while (Id_Mat>>m.ID>>m.matricula) {
           
           Id_MatriculaVector.push_back(m);
       }
        
    }else std::cout<<"Unable to open"<<id_matFile<<'\n';
}

Mat DataBase::getBiometricByMatricula(string matricula){
    Mat m;
    for(int i=0;i<Id_MatriculaVector.size();i++){
        if(Id_MatriculaVector[i].matricula == matricula){
            return descriptores.row(i);
        }
    }
    return m;
}

void DataBase::saveUserDataInAFile(BiographicalData bio){
    
    int id=n;
    std::cout<<"listo"<<std::endl;

    biographicalDB.open(biographicalFile,std::ios::out | std::ios::app);
    
    if(biographicalDB.is_open()){
        
        biographicalDB<<id<<","<<bio.matricula<<","<<bio.name<<","<<bio.lastName<<","<<bio.mail<<","<<bio.age<<"\n";
        biographicalDB.close();
    }else std::cout<<"Unable to open file\n";
    
}

void DataBase::saveUserBiometricDataInAFile(Mat biometric){
    string nuevoUsuario = "";
    int id=n;
    biometricDB.open(biometricFile,std::ios::out | std::ios::app);
    
    nuevoUsuario+=std::to_string(id);
    nuevoUsuario+=",";
    for(int i = 0 ; i< biometric.cols;i++){
        float nearest = roundf(biometric.at<float>(0,i) * 100) / 100;
        //cout<<nearest<<" ";
        nuevoUsuario+= std::to_string(nearest);
        if(i < biometric.cols-1){
            nuevoUsuario+=',';
        }
    }
    std::cout<<nuevoUsuario<<std::endl;
    biometricDB<<nuevoUsuario<<"\n";
}

void DataBase::getN(){
    N.open(nFile,std::ios::in);

    if(N.is_open()){
        N>>n;
        N.close();
    }else std::cout<<"Unable to open: "<<nFile<<'\n';
     
}

void DataBase::updateDataBase(){
    N.open(nFile);
    n=n+1;
    if(N.is_open()){
        std::cout<<"n: "<<n<<std::endl;
        N<<n<<"\n";
        N.close();
    }else std::cout<<"Error updating N.txt file\n";
   
}

void DataBase::saveUserImage(Mat &image){
    string fileNameLocation = "/Fotos/";
    fileNameLocation.append(std::to_string(n));
    fileNameLocation.append(".jpg");
    cv::imwrite(fileNameLocation, image);
}
