#include "facial_extractor_tools.h"
#include "Aligner.h"
#include "FaceLandmarkDetector.h"
#include "FaceAligner.h"

using cv::Mat;
using std::string;
using dlib::full_object_detection;
using cv::Size;
using dlib::rectangle;

void
example(int argc, char **argv, const string &path_to_image, const string &path_to_model, const cv::Rect &face,
        const unsigned int template_size, const float &left_eye) {
    Mat image;
    extractor::read_image(argc, argv, path_to_image,
                          &image); // by default

    FaceAligner face_transformer(path_to_model, template_size, left_eye);
    Mat template_image;
    face_transformer.DetectAndAlign(image, face, template_image);
    extractor::show_image(template_image);
}

int main(int argc, char **argv) {
    string path_to_image = "/home/axelzucho/Documents/Github/facial_transformation/images/Cara1.jpeg";
    string path_to_model = "/home/axelzucho/Documents/Github/facial_transformation/shape_predictor_5_face_landmarks.dat";
    cv::Rect face(330, 320, 310, 310);
    unsigned int template_size = 500;
    float left_eye = 0.3;
    example(argc, argv, path_to_image, path_to_model, face, template_size, left_eye);
    return 0;
}
