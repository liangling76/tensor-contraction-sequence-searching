cmake_minimum_required(VERSION 3.5)
set(CMAKE_CXX_STANDARD 11)
project(SeqSearch)

file(GLOB_RECURSE HEADER_LIST ${PROJECT_SOURCE_DIR}/include/*.h)
file(GLOB_RECURSE SRC_LIST ${PROJECT_SOURCE_DIR}/src/*.cpp)

include_directories(${PROJECT_SOURCE_DIR}/include)

SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
add_executable(SeqSearch "${HEADER_LIST}" "${SRC_LIST}")
TARGET_LINK_LIBRARIES(SeqSearch pthread)


