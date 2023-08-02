cmake_minimum_required(VERSION 3.15)
project(CUDA12-HOG VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()

find_package( OpenCV REQUIRED )
include_directories( ${OPENCV_INCLUDE_DIRS} )


add_executable(CUDA12-HOG 
    main.cpp
    hog_visualize.cpp
    hog_visualize.h)

target_link_libraries( CUDA12-HOG ${OpenCV_LIBS} )
target_link_libraries(CUDA12-HOG)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
