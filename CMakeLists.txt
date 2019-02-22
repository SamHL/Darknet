#=======================================================
# DARKNET CMAKE BUILD 
#=======================================================
cmake_minimum_required(VERSION 3.9)
PROJECT(DARKNET LANGUAGES C CXX)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

#=======================================================
# FIND DEPENDENCIES
#=======================================================
find_package(OpenCV REQUIRED HINTS "C:/Builds/Official/OpenCV4.0.1/Build")
message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

find_package(CUDA REQUIRED)
message(STATUS "CUDA library status:")
message(STATUS "    config: ${CUDA_TOOLKIT_ROOT_DIR}")
message(STATUS "    version: ${CUDA_VERSION}")
message(STATUS "    libraries: ${CUDA_LIBRARIES}")
message(STATUS "    include path: ${CUDA_INCLUDE_DIRS}")
message(STATUS "	checking for CUDNN in ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/...")
if(EXISTS "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cudnn.lib")
message(STATUS "	found cudnn")
set(CUDA_cudnn_LIBRARY "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cudnn.lib" CACHE FILEPATH "cudnn Library path")
else()
message("	could not find cudnn. Configure manually")
set(CUDA_cudnn_LIBRARY "" CACHE FILEPATH "cudnn Library path")
endif()
include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})


#=======================================================
# ADD PTHREADS LIBRARY AND TESTS
#=======================================================
	# Get the pthreads variables
set(PTHREADS_SRC_DIR "${PROJECT_SOURCE_DIR}/3rd Party/pthreads-win32")
include_directories(${PTHREADS_SRC_DIR})
add_subdirectory(${PTHREADS_SRC_DIR})
message(STATUS "PThreads library status:")
message(STATUS "    library build: ${PTHREADS_LIBRARY}")

#=======================================================
# ADD COMPILE DEFINITIONS
#=======================================================
add_compile_definitions(OPENCV)
add_definitions(-DGPU)
add_definitions(-DCUDNN)

#=======================================================
# ADD DARKNET LIBRARY
#=======================================================
file(GLOB darknetLibHeaders ${PROJECT_SOURCE_DIR}/include/*.h)
include_directories(${PROJECT_SOURCE_DIR}/include/)
file(GLOB darknetHeaders ${PROJECT_SOURCE_DIR}/src/*.h)
file(GLOB darknetSources ${PROJECT_SOURCE_DIR}/src/*.cpp)
file(GLOB darknetCUSources ${PROJECT_SOURCE_DIR}/src/*.cu)

CUDA_ADD_LIBRARY(DARKNET_LIB ${darknetSources} ${darknetHeaders} ${darknetCUSources} ${darknetLibHeaders} ${PTHREAD_API})
target_link_libraries(DARKNET_LIB ${OpenCV_LIBS} ${CUDA_cudnn_LIBRARY} ${CUDA_cublas_LIBRARY} ${CUDA_curand_LIBRARY} ${CUDA_LIBRARIES} pthreads)

#=======================================================
# ADD DARKNET EXAMPLE EXECUTABLE
#=======================================================
#[[
file(GLOB darknetExampleSources ${PROJECT_SOURCE_DIR}/examples/*.cpp)
CUDA_ADD_EXECUTABLE(Darknet_Examples ${darknetExampleSources} ${darknetLibHeaders} ${PTHREAD_API})
set_target_properties(Darknet_Examples PROPERTIES EXCLUDE_FROM_ALL TRUE)
target_link_libraries(Darknet_Examples ${OpenCV_LIBS} DARKNET_LIB pthreads ${OpenCV_LIBS})
]]
#=======================================================
# ADD YOLO TEST EXECUTABLE
#=======================================================
file(GLOB yoloExampleSources ${PROJECT_SOURCE_DIR}/yoloTest/*.cpp)
CUDA_ADD_EXECUTABLE(YOLO_TEST ${yoloExampleSources} ${darknetLibHeaders} ${PTHREAD_API})
target_link_libraries(YOLO_TEST ${OpenCV_LIBS} DARKNET_LIB)
