# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/axelzucho/Documents/dlib-19.15/dlib/cmake_utils/test_for_cuda

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/axelzucho/Documents/Github/facial_recognition/BUILD/dlib_build/cuda_test_build

# Include any dependencies generated for this target.
include CMakeFiles/cuda_test.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/cuda_test.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/cuda_test.dir/flags.make

CMakeFiles/cuda_test.dir/cuda_test_generated_cuda_test.cu.o: CMakeFiles/cuda_test.dir/cuda_test_generated_cuda_test.cu.o.depend
CMakeFiles/cuda_test.dir/cuda_test_generated_cuda_test.cu.o: CMakeFiles/cuda_test.dir/cuda_test_generated_cuda_test.cu.o.cmake
CMakeFiles/cuda_test.dir/cuda_test_generated_cuda_test.cu.o: /home/axelzucho/Documents/dlib-19.15/dlib/cmake_utils/test_for_cuda/cuda_test.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --progress-dir=/home/axelzucho/Documents/Github/facial_recognition/BUILD/dlib_build/cuda_test_build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building NVCC (Device) object CMakeFiles/cuda_test.dir/cuda_test_generated_cuda_test.cu.o"
	cd /home/axelzucho/Documents/Github/facial_recognition/BUILD/dlib_build/cuda_test_build/CMakeFiles/cuda_test.dir && /usr/bin/cmake -E make_directory /home/axelzucho/Documents/Github/facial_recognition/BUILD/dlib_build/cuda_test_build/CMakeFiles/cuda_test.dir//.
	cd /home/axelzucho/Documents/Github/facial_recognition/BUILD/dlib_build/cuda_test_build/CMakeFiles/cuda_test.dir && /usr/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING= -D generated_file:STRING=/home/axelzucho/Documents/Github/facial_recognition/BUILD/dlib_build/cuda_test_build/CMakeFiles/cuda_test.dir//./cuda_test_generated_cuda_test.cu.o -D generated_cubin_file:STRING=/home/axelzucho/Documents/Github/facial_recognition/BUILD/dlib_build/cuda_test_build/CMakeFiles/cuda_test.dir//./cuda_test_generated_cuda_test.cu.o.cubin.txt -P /home/axelzucho/Documents/Github/facial_recognition/BUILD/dlib_build/cuda_test_build/CMakeFiles/cuda_test.dir//cuda_test_generated_cuda_test.cu.o.cmake

# Object files for target cuda_test
cuda_test_OBJECTS =

# External object files for target cuda_test
cuda_test_EXTERNAL_OBJECTS = \
"/home/axelzucho/Documents/Github/facial_recognition/BUILD/dlib_build/cuda_test_build/CMakeFiles/cuda_test.dir/cuda_test_generated_cuda_test.cu.o"

libcuda_test.a: CMakeFiles/cuda_test.dir/cuda_test_generated_cuda_test.cu.o
libcuda_test.a: CMakeFiles/cuda_test.dir/build.make
libcuda_test.a: CMakeFiles/cuda_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --progress-dir=/home/axelzucho/Documents/Github/facial_recognition/BUILD/dlib_build/cuda_test_build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libcuda_test.a"
	$(CMAKE_COMMAND) -P CMakeFiles/cuda_test.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cuda_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/cuda_test.dir/build: libcuda_test.a

.PHONY : CMakeFiles/cuda_test.dir/build

CMakeFiles/cuda_test.dir/requires:

.PHONY : CMakeFiles/cuda_test.dir/requires

CMakeFiles/cuda_test.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/cuda_test.dir/cmake_clean.cmake
.PHONY : CMakeFiles/cuda_test.dir/clean

CMakeFiles/cuda_test.dir/depend: CMakeFiles/cuda_test.dir/cuda_test_generated_cuda_test.cu.o
	cd /home/axelzucho/Documents/Github/facial_recognition/BUILD/dlib_build/cuda_test_build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/axelzucho/Documents/dlib-19.15/dlib/cmake_utils/test_for_cuda /home/axelzucho/Documents/dlib-19.15/dlib/cmake_utils/test_for_cuda /home/axelzucho/Documents/Github/facial_recognition/BUILD/dlib_build/cuda_test_build /home/axelzucho/Documents/Github/facial_recognition/BUILD/dlib_build/cuda_test_build /home/axelzucho/Documents/Github/facial_recognition/BUILD/dlib_build/cuda_test_build/CMakeFiles/cuda_test.dir/DependInfo.cmake
.PHONY : CMakeFiles/cuda_test.dir/depend

