cmake_minimum_required( VERSION 3.11 )

# set version and project
set(etagger_VERSION_MAJOR 1)
set(etagger_VERSION_MINOR 0)
set(etagger_VERSION_PATCH 0)
set(etagger_VERSION ${etagger_VERSION_MAJOR}.${etagger_VERSION_MINOR}.${etagger_VERSION_PATCH})
if (POLICY CMP0048)
  cmake_policy(SET CMP0048 NEW)
endif (POLICY CMP0048)
project(etagger VERSION ${etagger_VERSION})

# check dependencies
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../cmake/modules)
find_package(TensorFlow 1.11 EXACT REQUIRED)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=${TensorFlow_ABI}")
set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=${TensorFlow_ABI}" )
TensorFlow_REQUIRE_C_LIBRARY()
TensorFlow_REQUIRE_SOURCE()

# for linking libtensorflow_cc.so, libtensorflow_framework.so in the ${TENSORFLOW_BUILD_DIR}
find_library(TensorFlow_CC_LIBRARY
  NAMES libtensorflow_cc.so
  PATHS $ENV{TENSORFLOW_BUILD_DIR}
  DOC "TensorFlow C library." )
add_library(TensorFlow_CC_DEP INTERFACE)
TARGET_LINK_LIBRARIES(TensorFlow_CC_DEP INTERFACE -Wl,--allow-multiple-definition -Wl,--whole-archive ${TensorFlow_CC_LIBRARY} -Wl,--no-whole-archive)

find_library(TensorFlow_FW_LIBRARY
  NAMES libtensorflow_framework.so
  PATHS $ENV{TENSORFLOW_BUILD_DIR}
  DOC "TensorFlow framework library." )
add_library(TensorFlow_FW_DEP INTERFACE)
TARGET_LINK_LIBRARIES(TensorFlow_FW_DEP INTERFACE -Wl,--allow-multiple-definition -Wl,--whole-archive ${TensorFlow_FW_LIBRARY} -Wl,--no-whole-archive)

# build libraries
include_directories(include)
set(etagger_src "src/Config.cc" "src/Vocab.cc" "src/Input.cc" "src/TFUtil.cc" "src/Etagger.cc")
add_library(etagger SHARED ${etagger_src})
target_include_directories(etagger PRIVATE TensorFlow_DEP)
target_link_libraries(etagger PRIVATE TensorFlow_DEP)
target_link_libraries(etagger PRIVATE TensorFlow_FW_DEP)
target_link_libraries(etagger PRIVATE TensorFlow_CC_DEP)
set_target_properties(etagger PROPERTIES VERSION ${etagger_VERSION}
                                         SOVERSION ${etagger_VERSION_MAJOR})
add_library(etagger_static STATIC ${etagger_src})
target_include_directories(etagger_static PRIVATE TensorFlow_DEP)
target_link_libraries(etagger_static PRIVATE TensorFlow_DEP)
target_link_libraries(etagger_static PRIVATE TensorFlow_CC_DEP)
target_link_libraries(etagger_static PRIVATE TensorFlow_FW_DEP)

# build executable binaries
add_executable (inference src/inference.cc)
target_include_directories(inference PRIVATE TensorFlow_DEP)
target_link_libraries(inference PRIVATE TensorFlow_DEP)
target_link_libraries(inference PRIVATE TensorFlow_CC_DEP)
target_link_libraries(inference PRIVATE TensorFlow_FW_DEP)
target_link_libraries(inference PRIVATE etagger)


# build misc executable binaries
add_executable (inference_example src/inference_example.cc)
target_include_directories(inference_example PRIVATE TensorFlow_DEP)
target_link_libraries(inference_example PRIVATE TensorFlow_DEP)
target_link_libraries(inference_example PRIVATE TensorFlow_CC_DEP)
target_link_libraries(inference_example PRIVATE TensorFlow_FW_DEP)

add_executable (inference_iris src/inference_iris.cc)
target_include_directories(inference_iris PRIVATE TensorFlow_DEP)
target_link_libraries(inference_iris PRIVATE TensorFlow_DEP)
target_link_libraries(inference_iris PRIVATE TensorFlow_CC_DEP)
target_link_libraries(inference_iris PRIVATE TensorFlow_FW_DEP)

