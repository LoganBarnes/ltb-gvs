##########################################################################################
# LTB Geometry Visualization Server
# Copyright (c) 2020 Logan Barnes - All Rights Reserved
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
##########################################################################################
cmake_minimum_required(VERSION 3.15)
project(LtbGeometryVisualizationServer LANGUAGES CXX)

option(LTB_ENABLE_GVS_DISPLAY "Build the GUI targets that display scenes" ON)
option(LTB_ENABLE_GVS_NETWORKING "Build the networking targets that enable remote scene manipulation" ON)
option(LTB_BUILD_GVS_EXAMPLES "Build example programs" OFF)

include(ltb-util/cmake/LtbConfig.cmake) # <-- Additional project options are in here.

############
### Util ###
############
add_subdirectory(ltb-util)

########################
### External Targets ###
########################
set(LTB_GVS_ROOT ${CMAKE_CURRENT_LIST_DIR})

include(cmake/ThirdPartyCore.cmake)

if (LTB_ENABLE_GVS_DISPLAY)
    include(cmake/ThirdPartyGraphics.cmake)
    include(cmake/ThirdPartyGui.cmake)
    include(cmake/ThirdPartyCuda.cmake)

    ### GVS Display Resources ###
    # Compile resources separately to avoid linting issues on generated code
    corrade_add_resource(ltb_gvs_display_RESOURCES ${CMAKE_CURRENT_LIST_DIR}/src/resources.conf)
    add_library(ltb_gvs_display_resources ${ltb_gvs_display_RESOURCES})
    target_link_libraries(ltb_gvs_display_resources
            PUBLIC
            LtbExternal::Graphics
            )
endif (LTB_ENABLE_GVS_DISPLAY)

if (LTB_ENABLE_GVS_NETWORKING)
    include(cmake/ThirdPartyNetworking.cmake)
endif (LTB_ENABLE_GVS_NETWORKING)

#####################
### Configuration ###
#####################
configure_file(
        ${CMAKE_CURRENT_LIST_DIR}/src/gvs_paths.hpp.in
        ${CMAKE_BINARY_DIR}/generated/ltb/gvs_paths.hpp
)

configure_file(
        ${CMAKE_CURRENT_LIST_DIR}/src/gvs_cuda_includes.hpp.in
        ${CMAKE_BINARY_DIR}/generated/ltb/gvs_cuda_includes.hpp
)

####################
### Source Files ###
####################
file(GLOB_RECURSE LTB_CORE_SOURCE_FILES
        LIST_DIRECTORIES false
        CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_LIST_DIR}/src/ltb/gvs/core/*
        )

file(GLOB_RECURSE LTB_TEST_SOURCE_FILES
        LIST_DIRECTORIES false
        CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_LIST_DIR}/src/ltb/testing/*
        )

file(GLOB_RECURSE LTB_GRPCW_SOURCE_FILES
        LIST_DIRECTORIES false
        CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_LIST_DIR}/src/ltb/grpcw/*
        )

file(GLOB_RECURSE LTB_DISPLAY_SOURCE_FILES
        LIST_DIRECTORIES false
        CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_LIST_DIR}/src/ltb/gvs/display/*
        )

file(GLOB_RECURSE LTB_OPTIX_SOURCE_FILES
        LIST_DIRECTORIES false
        CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_LIST_DIR}/src/ltb/cuda/optix/*
        )

file(GLOB_RECURSE LTB_CUDA_SOURCE_FILES
        LIST_DIRECTORIES false
        CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_LIST_DIR}/src/ltb/cuda/*
        )
list(REMOVE_ITEM LTB_CUDA_SOURCE_FILES ${LTB_OPTIX_SOURCE_FILES})

###########
### GVS ###
###########
ltb_add_library(ltb_gvs
        17
        ${LTB_CORE_SOURCE_FILES}
        $<$<BOOL:${LTB_ENABLE_GVS_NETWORKING}>:${LTB_GRPCW_SOURCE_FILES}>
        $<$<BOOL:${LTB_ENABLE_GVS_DISPLAY}>:${LTB_DISPLAY_SOURCE_FILES}>
        $<$<BOOL:${CUDAToolkit_FOUND}>:${LTB_CUDA_SOURCE_FILES}>
        $<$<BOOL:${OptiX_INCLUDE}>:${LTB_OPTIX_SOURCE_FILES}>
        $<$<AND:$<BOOL:${LTB_ENABLE_GVS_DISPLAY}>,$<BOOL:${LTB_BUILD_TESTS}>>:${LTB_TEST_SOURCE_FILES}>
        )

ltb_include_directories(ltb_gvs
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/src>
        )

ltb_link_libraries(ltb_gvs
        PUBLIC
        LtbExternal::Doctest
        LtbExternal::BoostUuid
        LtbExternal::Mapbox
        $<$<OR:$<BOOL:${LTB_ENABLE_GVS_DISPLAY}>,$<BOOL:${LTB_ENABLE_GVS_NETWORKING}>>:ltb_util>
        $<$<BOOL:${LTB_ENABLE_GVS_DISPLAY}>:ltb_gvs_display_resources>
        $<$<BOOL:${LTB_ENABLE_GVS_DISPLAY}>:ltb_external_gui>
        $<$<BOOL:${CUDAToolkit_FOUND}>:ltb_external_cuda>
        )

add_library(Ltb::Gvs ALIAS ltb_gvs)

if (LTB_BUILD_GVS_EXAMPLES)
    add_subdirectory(examples)
endif ()


##################################
### GVS C++11 Compilation Test ###
##################################
ltb_add_library(ltb_gvs_core_compilation 11 ${LTB_CORE_SOURCE_FILES})
ltb_link_libraries(ltb_gvs_core_compilation
        PUBLIC
        LtbExternal::BoostUuid
        LtbExternal::Mapbox
        )
ltb_include_directories(ltb_gvs_core_compilation
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/src>
        )
