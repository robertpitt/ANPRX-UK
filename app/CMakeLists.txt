# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.4.1 FATAL_ERROR)

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.

set(CMAKE_VERBOSE_MAKEFILE on)
set(OpenCV_DIR "../vendor/OpenCV-android-sdk/sdk/native/jni")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

ADD_LIBRARY(OpenCV SHARED IMPORTED REQUIRED)
ADD_LIBRARY(opencv_anpr SHARED src/main/cpp/opencv_anpr/opencv_anpr.cpp)

TARGET_LINK_LIBRARIES(opencv_anpr ${OpenCV_LIBS})