# idf/demo/CMakeLists.txt — minimal ESP-IDF project for the Ferrite component.
#
#   cd idf/demo
#   idf.py set-target esp32s3
#   idf.py build flash monitor
cmake_minimum_required(VERSION 3.16)

set(EXTRA_COMPONENT_DIRS "${CMAKE_CURRENT_LIST_DIR}/../ferrite")

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(ferrite_demo)