#
# This is a CMake makefile.  You can find the cmake utility and
# information about it at http://www.cmake.org
#

cmake_minimum_required(VERSION 3.1)
project(fr)

set(CMAKE_CXX_STANDARD 11)


include(/home/carrasc/dlib-19.16/dlib/cmake)



set(HEADERS FaceAligner/facial_extractor_tools.h FaceAligner/Aligner.h FaceAligner/FaceLandmarkDetector.h FaceAligner/FaceAligner.h FaceDescriptorExtractor/FaceDescriptorExtractor.h FaceDetection/FaceDetector_opt.h DB/DataBase.h FaceRecognition.h FaceRecognitionTools.h
        )

set(SOURCE FaceAligner/facial_extractor_tools.cpp FaceAligner/Aligner.cpp FaceAligner/FaceLandmarkDetector.cpp FaceAligner/FaceAligner.cpp FaceDescriptorExtractor/FaceDescriptorExtractor.cpp FaceDetection/FaceDetector_opt.cpp DB/DataBase.cpp FaceRecognition.cpp FaceRecognitionTools.cpp
        )

# automatic OpenCV finding
find_package(OpenCV REQUIRED)

# if not, use manual specification of the OpenCVConfig.cmake path
# and comment previous line of automatic finding
#find_package(OpenCV REQUIRED PATHS /usr/local/share/OpenCV NO_DEFAULT_PATH)


if (OpenCV_FOUND)
    message("-- Opencv Found")
endif ()

add_definitions(-DLIB_USE_AVX)

set(fr_include /usr/local/include
        )

include_directories(${fr_include})

add_executable(fr main.cpp ${HEADERS} ${SOURCE})
target_link_libraries(fr ${OpenCV_LIBS} dlib::dlib)
