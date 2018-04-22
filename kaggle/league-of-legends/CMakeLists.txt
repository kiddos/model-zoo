cmake_minimum_required(VERSION 3.0)
include(ExternalProject)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_CXX_FLAGS
  "${CMAKE_CXX_FLAGS} -fPIC -Wall -Wextra -Wno-unused-parameter")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")

project(lol)

find_library(SQLITE_LIBRARY sqlite3)
if (NOT SQLITE_LIBRARY)
  message(FATAL_ERROR "fail to find sqlite library")
else ()
  message(STATUS "sqlite: ${SQLITE_LIBRARY}")
endif ()

find_library(GLOG_LIBRARY glog)
if (NOT GLOG_LIBRARY)
  message(FATAL_ERROR "fail to find glog library")
else ()
  message(STATUS "glog: ${GLOG_LIBRARY}")
endif ()

find_package(Protobuf REQUIRED)
include_directories(${PROTOBUF_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS proto/league.proto)

execute_process(COMMAND protoc -I ${CMAKE_SOURCE_DIR}/proto
  ${CMAKE_SOURCE_DIR}/proto/league.proto
  --grpc_out=${CMAKE_BINARY_DIR}
  --plugin=protoc-gen-grpc=/usr/local/bin/grpc_cpp_plugin)

find_library(GRPC_LIBRARY grpc)
if (NOT GRPC_LIBRARY)
  message(FATAL_ERROR "fail to find grpc library")
else ()
  message(STATUS "grpc: ${GRPC_LIBRARY}")
endif ()

find_library(GRPCPP_LIBRARY grpc++)
if (NOT GRPCPP_LIBRARY)
  message(FATAL_ERROR "fail to find grpc++ library")
else ()
  message(STATUS "grpc++: ${GRPCPP_LIBRARY}")
endif ()

find_package(Qt5Widgets REQUIRED)
include_directories(${CMAKE_BINARY_DIR} ${Qt5Widgets_INCLUDE_DIRS})
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_CXX_FLAGS_COVERAGE "${CMAKE_CXX_FLAGS_RELEASE} -fprofile-arcs -ftest-coverage")
get_target_property(QtWidget_location Qt5::Widgets LOCATION)

message(STATUS "QtWidget Version: ${Qt5Widgets_VERSION} at ${QtWidget_location}")

qt5_wrap_ui(PREDICT_WINDOW_HEADER predict.ui)
file(GLOB CPP_SOURCE *.cc)
file(GLOB GRPCPP_SOURCE ${CMAKE_BINARY_DIR}/league.grpc.pb.*)
add_executable(predict
  ${CPP_SOURCE}
  ${PROTO_SRCS}
  ${PROTO_HDRS}
  ${GRPCPP_SOURCE}
  ${PREDICT_WINDOW_HEADER})

target_link_libraries(predict
  Qt5::Widgets
  ${GLOG_LIBRARY}
  ${SQLITE_LIBRARY}
  ${PROTOBUF_LIBRARIES}
  ${GRPC_LIBRARY}
  ${GRPCPP_LIBRARY})

file(COPY ${CMAKE_SOURCE_DIR}/league.sqlite3 DESTINATION ${CMAKE_BINARY_DIR})
