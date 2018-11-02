# Face Aligner
This repository will provide a way of transforming an already detected face and output it in a predefined template.

## Class

### Constructor

The constructor for the FaceAligner class needs three arguments:
1. String: Path for the landmark model (See FaceLandmarkDetector class).
2. Int: Size of the template image where the aligned face will be shown.
3. Float: Value from 0 to 1 giving the x and y coordinate of the left eye in the template image.

### Functions

1. Detect: Detects the landmarks for the given image:
   1. cv::Mat: Image to detect
   2. cv::Rect: Rectangle that bounds the face
   3. dlib::full_object_detection: Struct where the detected landmark points will be saved.

2. Align: Aligns the given image according to the points given and outputs it to the given image:
   1. dlib::full_object_detection: Detected points used to align image
   2. cv::Mat: Image to align
   3. cv::Mat: Image to output the aligned face

3. DetectAndAlign: Detects the landmarks for the given image and face, aligns the image and outputs it to the given Mat
   1. cv::Mat: Image to detect and align
   2. cv::Rect: Rectangle that bounds the face
   3. cv::Mat: Image to output the aligned face


## FaceLandmarkDetector

This class will detect the landmarks for the given face and the given image.

### Constructor

The constructor for this class needs the path for the landmark detector model to use. There are two in this repository:
1. shape_predictor_5_face_landmarks.dat
2. shape_predictor_68_face_landmarks.dat

Both of these were downloaded from [this repository](https://github.com/davisking/dlib-models). We recomend the use of the shape_predictor_5_face_landmarks.

### Functions

1. GetFaceLandmark: Detects the important points of a given image.
   1. cv::Mat: Image to detect
   2. cv::Rect: Rectangle that bounds the face
   3. dlib::full_object_detection: Struct where the detected landmark points will be saved.
   

## Aligner

This class will align an image based on given points.

### Constructor

The constructor of this class needs two arguments:
1. Int: Size of the template image where the aligned face will be shown.
2. Float: Value from 0 to 1 giving the x and y coordinate of the left eye in the template image.

### Functions 

1. AlignImage: Aligns a given image based on the points given and outputs it to the given image.
   1. dlib::full_object_detection: Detected points used to align image
   2. cv::Mat: Image to align
   3. cv::Mat: Image to output the aligned face

