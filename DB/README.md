# DB
This repository will provide a way of creating, storing, and accessing a datbase of biometric and biographical features.

## Constructor

The default constructor requires no arguments and will assume all the files are within the running directory.

The second constructor requiress four arguments:
1. String: The path to the .txt containing biographical data.
2. String: The path to the .txt containing biometric data.
3. String: The path to the .txt containing the current ID number.
4. String: The path to the .txt relating internal IDs to external IDs (matricula).

## Functions

The class' functions are listed below with their respective parameters listed below them if applicable.

### File management

1. load_N_File, loads N.txt

2. load_ImgFolder, creates a directory for images

3. load_BiometricFile, loads biometrics.txt

4. load_BiographicalFile, loads BiographicalData.txt

5. load_Id_MatriculaFile, loads ID_mat.txt

### Data management

6. Mat getMatrix, retrieve entire biometric matrix

7. Mat getColumn
   1. int, the column to be returned by the function

8. Mat getRow
   1. int, the row to be returned by the function

9. std::pair <Mat, Mat> search, returns a matrix of similar vectors
   1. Mat, the vector to be searched
   2. int, the number of similar vectors to be returned

10. Mat getBiometricByMatricula, returns a biometric vector of a given person
   1. string, the ID (matricula) to be searched

11. BiographicalData getUserInfoByID, returns a data type BiographicalData containing biographical data for a given person
   1. int, the ID (given and used by the system) to be searched

12. saveUserDataInAFile, saves user biographical data in the biographical data txt file
   1. BiographicalData, a structure in which biographical data is contained

13. saveUserBiometricDataInAFile, saves user biometric data in the biometric data txt file
   1. Mat, a vector in which biometric data is contained

14. saveUserImage, saves the image of the user
   1. Mat, the image of the user, uncropped and unaligned

15. getN, updates the N.txt

16. updateDataBase, updates the biometric database file (changes will be made only within the program's Mat and will be lost on closing unless this function is ran)

17. std::vector<std::string> indexData, returns a vector of biographical data
   1. std::string, a string of biographical data, comma delimited

18. BiographicalData String_To_Structure, returns a BiographicalData structure
   1. std::string, a string of biographical data, comma delimited

19. bool ValidName, returns true if a name contains only valid characters
   1. std::string, the name to be validated

20. bool SimpleValidateMail, returns true if an email is valid
   1. std::string, the email

21. int ValidateData, returns an error code if there are any issues with a user's biographical data (name, email, ID (matricula)
   1. BiographicalData*, a pointer to the BiographicalData structure containing the biographical data
