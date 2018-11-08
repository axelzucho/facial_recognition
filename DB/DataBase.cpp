#include "DataBase.h"
#include "opencv2/opencv.hpp"


DataBase::DataBase(){
    this->biographicalFile= "../BiographicalData.txt";
    this->biometricFile = "../biometrics.txt";
    this->nFile = "../N.txt";
    this->id_matFile = "../ID_mat.txt";
    
    load_N_File();
    load_ImgFolder();
    
    if(existsFile(biometricFile)){
    
        load_BiometricFile();
        //load_BiographicalFile();
        load_Id_MatriculaFile();
    
        flann_index = new Index(descriptores, cv::flann::KDTreeIndexParams());
        std::cout<<"listo"<<std::endl;
    }
}

DataBase::DataBase(string biographicalFile,string biometricFile,string nFile,string id_matFile){
    this->biographicalFile=biographicalFile;
    this->biometricFile=biometricFile;
    this->nFile=nFile;
    this->id_matFile=id_matFile;

    
    load_N_File();
    load_ImgFolder();
    
    if(existsFile(biometricFile)){
        
        load_BiometricFile();
        //load_BiographicalFile();
        load_Id_MatriculaFile();
        
        flann_index = new Index(descriptores, cv::flann::KDTreeIndexParams());
        std::cout<<"listo"<<std::endl;
    }
}
void DataBase::load_N_File(){
    
    if(!existsFile(nFile)){
        N.open(nFile,std::ios::out);
        if(N.is_open()){
            N<<"0\n";
            std::cout<<"***Fichero N.txt creado con éxito***\n";
            N.close();
        }else std::cout<< "Unable creating N.txt\n";
    }
}
inline bool DataBase::existsFile (const std::string& name) {
    std::ifstream f(name.c_str());
    return f.good();
    
}
void DataBase::load_ImgFolder(){
    if(mkdir("../Img", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == 0){
        std::cout<<"Directory successfully created"<<'\n';
        
    }else std::cout<<"Error creating directory or the directory already exits"<<'\n';
}


void DataBase::load_BiographicalFile(){
    
    string line;
    biographicalDB.open(biographicalFile, std::ios::in);
    
    if(biographicalDB.is_open()){
        while (true) {
            std::getline(biographicalDB,line);
            
            biograData.push_back(String_To_Structure(line));
            if(!biographicalDB.eof()){
                break;
            }
        }
    }else std::cout<<"Unable to open: "<<biographicalFile<<'\n';
}

void DataBase::load_BiometricFile(){
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
void DataBase::load_Id_MatriculaFile(){
    MatriculaId m;
    
    Id_Mat.open(id_matFile,std::ios::in);
    
    if(Id_Mat.is_open()){
        while (Id_Mat>>m.ID>>m.matricula) {
            
            Id_MatriculaVector.push_back(m);
        }
        
    }else std::cout<<"Unable to open: "<<id_matFile<<'\n';
}

Mat DataBase::getBiometricByMatricula(string matricula){
    Mat m;
    for(int i=0;i<Id_MatriculaVector.size();i++){
        if(Id_MatriculaVector[i].matricula == matricula){
            return descriptores.row(i);
        }
    }
    return m.clone();
}

BiographicalData DataBase::getUserInfoByID(int ID){
    return biograData[ID];
}

void DataBase::saveUserDataInAFile(BiographicalData bio){
    
    int id=n;

    biographicalDB.open(biographicalFile,std::ios::out | std::ios::app);
    
    if(biographicalDB.is_open()){
        
        biographicalDB<<id<<","<<bio.matricula<<","<<bio.name<<","<<bio.lastName<<","<<bio.mail<<","<<bio.age<<","<<bio.img<<"\n";
        biographicalDB.close();
    }else std::cout<<"Unable to open file: "<<biographicalFile<<'\n';
    
    Id_Mat.open(id_matFile,std::ios::out | std::ios::app);
    
    if(Id_Mat.is_open()){
        
        Id_Mat<<id<<" "<<bio.matricula<<"\n";
        Id_Mat.close();
    }else std::cout<<"Unable to open file: "<<id_matFile<<'\n';
    
}

void DataBase::saveUserBiometricDataInAFile(Mat biometric){
    string nuevoUsuario = "";
    int id=n;
    biometricDB.open(biometricFile,std::ios::out | std::ios::app);
    
    nuevoUsuario+=std::to_string(id);
    nuevoUsuario+=",";
    for(int i = 0 ; i< biometric.rows;i++){
        float nearest = roundf(biometric.at<float>(i,0) * 100) / 100;
        //cout<<nearest<<" ";
        nuevoUsuario+= std::to_string(nearest);
        if(i < biometric.rows-1){
            nuevoUsuario+=',';
        }
    }
    //std::cout<<nuevoUsuario<<std::endl;
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
        //std::cout<<"n: "<<n<<std::endl;
        N<<n<<"\n";
        N.close();
    }else std::cout<<"Error updating N.txt file\n";
   
}

void DataBase::saveUserImage(Mat &image){
    //cv::imwrite("Fotos/2.jpg",image);
    string fileNameLocation = "../Img/";
    fileNameLocation.append(std::to_string(n));
    fileNameLocation.append(".jpg");
    cv::imwrite(fileNameLocation, image);
}
//Splits the dataLine (strings) to a vector of strings
std::vector<std::string> DataBase::indexData(std::string dataLine) {
	std::vector <int> keys;
	std::vector<std::string> dataMatrix;
	std::string line = dataLine;					//Holds one line of data
	std::string delimiter = ",";						//The character separating the data
	size_t pos = 0;								//Starting at index 0
	int j = 0;
	std::string data;
	while ((pos = line.find(delimiter)) != std::string::npos) {	//While the delimiter still exits
		data = line.substr(0, pos);							//Get one piece of data (up to the delimiter)
		dataMatrix.push_back(data);			//String to float of bit of isolated data, add to data matrix
		line.erase(0, pos + delimiter.length());			//Erase the line up to the delimiter
		j++;
	}
	dataMatrix.push_back(line);			//Last bit of data doesn't have a delimiter
	return dataMatrix;
}

BiographicalData DataBase::String_To_Structure(std::string Data_As_String)
	{
		BiographicalData Data_Struct;
		std::vector<std::string> Data_Vector_String=indexData(Data_As_String);
		Data_Struct.id=Data_Vector_String[0];
		Data_Struct.matricula=Data_Vector_String[1];
		Data_Struct.name=Data_Vector_String[2];
		Data_Struct.lastName=Data_Vector_String[3];
		Data_Struct.mail=Data_Vector_String[4];
		Data_Struct.age=std::stoi(Data_Vector_String[5]);
		Data_Struct.img=Data_Vector_String[6];
		return Data_Struct;
	}
int DataBase::ValidateData(BiographicalData *bio)
{
    int result_case_3=0;
    if(bio->matricula.length()!=9||bio->matricula[0]!='A')
        result_case_3+=1;
    if(!ValidName(bio->name))
        result_case_3+=2;
    if(!ValidName(bio->lastName))
        result_case_3+=4;
    if(!SimpleValidateMail(bio->mail))
        result_case_3+=8;
    if(bio->age>100||bio->age<1)
        result_case_3+=16;
    return result_case_3;
}
bool DataBase::ValidName(std::string word)
{
    bool verdict=1;
    //std::string Caps="ABCDEFGHIJKLMN�OPQRSTUVWXYZ�����";
    //std::string Lows="abcdefghijklmn�opqrstuvwxyz�����";
    std::string InvalidNameCharacters=" !\"#$%&()*+,./:;<=>?[\\]^_{|}~";
    for(int H=0;H<word.size();H++)
    {
        if(isdigit(word[H])||InvalidNameCharacters.find(word[H])!=std::string::npos)
        {
            verdict=0;
            break;
        }
    }
        return verdict;
}	
bool DataBase::SimpleValidateMail(std::string mail)
{
    if(mail.find('@')==std::string::npos)
        return false;
    else
    {
        if(mail.substr(mail.find('@'),mail.size()-mail.find('@')).find('.')==std::string::npos)
            return false;
        else
            return true;
    }
}

	

