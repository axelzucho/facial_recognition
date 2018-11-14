# Facial Recognition
This facial recognition project is an integration of a joint work between different teams.

### Modules

The project has four modules:
1. Face Detection: Detects the largest face of a given image and returns its location.
2. Face Aligner: Detects the important landmark points of a given face and returns an aligned face.
3. Face Descriptor Extractor: Describes the features of a given aligned image so that it can recognize different images.
4. DB: Saves and fetches information of given features to be used to recognize a certain person.

### Cases

This project allows to perform three different cases:

1. Validate: Compares a taken picture with the descriptors of a saved picture in the DB for a given ID. Returns true if the person is recognized as the same one from the DB.
2. Recognize: Gives the x amount of people with the highest amount of similarity between all the people saved in the DB.
3. Register: Saves the given person, with its picture, ID, and other relevant data into the DB.

#### Tools 

* Dlib library: http://dlib.net/
* Open CV
* Alignment models downloaded from [this repository](https://github.com/davisking/dlib-models). 
