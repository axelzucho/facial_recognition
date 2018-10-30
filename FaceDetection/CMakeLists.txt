set (CMAKE_CXX_STANDARD 11)

cmake_minimum_required(VERSION 2.8)
project( imgLoadnDisp )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( main main.cpp FaceDetector_opt.h FaceDetector_opt.cpp)
target_link_libraries( main ${OpenCV_LIBS} )
