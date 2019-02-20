cmake_minimum_required( VERSION 2.8 )

project( TFCustomOp )

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../cmake/modules)
find_package(TensorFlow 1.11 EXACT REQUIRED)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=${TensorFlow_ABI}")
set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=${TensorFlow_ABI}" )

TensorFlow_REQUIRE_C_LIBRARY()
TensorFlow_REQUIRE_SOURCE()

# for linking libtensorflow_framework.so in the ${TENSORFLOW_BUILD_DIR}
find_library(TensorFlow_FW_LIBRARY
  NAMES libtensorflow_framework.so
  PATHS $ENV{TENSORFLOW_BUILD_DIR}
  DOC "TensorFlow framework library." )
add_library(TensorFlow_FW_DEP INTERFACE)
TARGET_LINK_LIBRARIES(TensorFlow_FW_DEP INTERFACE -Wl,--allow-multiple-definition -Wl,--whole-archive ${TensorFlow_FW_LIBRARY} -Wl,--no-whole-archive)

add_executable (inference_example src/inference_example.cc)
target_include_directories(inference_example PRIVATE TensorFlow_DEP)
target_link_libraries(inference_example PRIVATE TensorFlow_DEP)
target_link_libraries(inference_example PRIVATE TensorFlow_FW_DEP)

add_executable (inference_iris src/inference_iris.cc)
target_include_directories(inference_iris PRIVATE TensorFlow_DEP)
target_link_libraries(inference_iris PRIVATE TensorFlow_DEP)
target_link_libraries(inference_iris PRIVATE TensorFlow_FW_DEP)

include_directories(include)
add_executable (inference src/inference.cc src/Config.cc src/Vocab.cc src/Input.cc src/TFUtil.cc)
target_include_directories(inference PRIVATE TensorFlow_DEP)
target_link_libraries(inference PRIVATE TensorFlow_DEP)
target_link_libraries(inference PRIVATE TensorFlow_FW_DEP)

add_executable (inference_mm src/inference_mm.cc src/Config.cc src/Vocab.cc src/Input.cc src/TFUtil.cc)
target_include_directories(inference_mm PRIVATE TensorFlow_DEP)
target_link_libraries(inference_mm PRIVATE TensorFlow_DEP)
target_link_libraries(inference_mm PRIVATE TensorFlow_FW_DEP)
