#
# This is a CMake makefile.  You can find the cmake utility and
# information about it at http://www.cmake.org
#

cmake_minimum_required (VERSION 3.1)
project(fr)

set (CMAKE_CXX_STANDARD 11)

include(/home/axelzucho/Documents/dlib-19.15/dlib/cmake)

 set(HEADERS facial_extractor_tools.h Aligner.h FaceLandmarkDetector.h FaceAligner.h
    )

set(SOURCE	facial_extractor_tools.cpp Aligner.cpp FaceLandmarkDetector.cpp FaceAligner.cpp
		)

# automatic OpenCV finding
find_package(OpenCV REQUIRED) 

# if not, use manual specification of the OpenCVConfig.cmake path
# and comment previous line of automatic finding
#find_package(OpenCV REQUIRED PATHS /usr/local/share/OpenCV NO_DEFAULT_PATH) 


if(OpenCV_FOUND)
	message("-- Opencv Found")
endif()

add_definitions(-DLIB_USE_AVX)

set(fr_include 	/usr/local/include 
	)

include_directories(${fr_include})

add_executable(fr main.cpp ${HEADERS} ${SOURCE})
target_link_libraries(fr ${OpenCV_LIBS} dlib::dlib)
