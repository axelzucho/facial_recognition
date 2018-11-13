# CLASS
## FaceDescriptorExtractor
Modulo 3 para el reconocimiento facial
This module will provide functions to extract and compare vectorial descriptors of faces.


### Constructor
  The constructor by default will provide a path to the model already trained that will be used to obtain the face descriptors.
  You can also load the class providing a different path in a string, that contains the file of the model.
  
### Functions
  1. obtenerDescriptorVectorial(cv::Mat given) : this function will return a cv::Mat that contains the vectorial descriptor that represents the face given:
        1. cv::Mat given - this Mat contains the image that we will analize and obtain the vectorial descriptors.
        2. cv::Mat result - this Mat contains the descriptors of the Mat given and we return it.
        
  2. compararDescriptores(cv::Mat 1, cv::Mat 2) : this function returns the euclidean distance as a float:
        1. cv::Mat 1 - contains the descriptors that will be compared against Mat 2.
        2. cv::Mat 2 - "                                                     " Mat 1.
        3. float - we return the euclidean distance, if there is any error we return a negative number
