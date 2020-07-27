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
option(LTB_BUILD_GVS_EXAMPLES "Build example programs" OFF)

include(ltb-util/cmake/LtbConfig.cmake) # <-- Additional project options are in here.

set(LTB_GVS_PROJECT_ROOT ${CMAKE_CURRENT_LIST_DIR})

###########
### GVS ###
###########
add_library(ltb_gvs INTERFACE)
add_library(Ltb::Gvs ALIAS ltb_gvs)

configure_file(
        ${CMAKE_CURRENT_LIST_DIR}/src/gvs_paths.hpp.in
        ${CMAKE_BINARY_DIR}/generated/ltb/gvs_paths.hpp
)

################
### GVS Core ###
################
include(cmake/ThirdPartyCore.cmake)

file(GLOB_RECURSE LTB_SOURCE_FILES
        LIST_DIRECTORIES false
        CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_LIST_DIR}/src/ltb/gvs/core/*
        )

ltb_add_library(ltb_gvs_core 11 ${LTB_SOURCE_FILES})
ltb_link_libraries(ltb_gvs_core
        PUBLIC
        LtbExternal::BoostUuid
        LtbExternal::Mapbox
        )
ltb_include_directories(ltb_gvs_core
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/src>
        )

add_library(LtbGvs::Core ALIAS ltb_gvs_core)

target_link_libraries(ltb_gvs INTERFACE ltb_gvs_core)

if (LTB_ENABLE_GVS_DISPLAY)
    include(cmake/ThirdPartyGraphics.cmake)
    include(cmake/ThirdPartyGui.cmake)

    ############
    ### Util ###
    ############
    add_subdirectory(ltb-util)

    ###############
    ### Testing ###
    ###############
    if (LTB_BUILD_TESTS)
        file(GLOB_RECURSE LTB_SOURCE_FILES
                LIST_DIRECTORIES false
                CONFIGURE_DEPENDS
                ${CMAKE_CURRENT_LIST_DIR}/src/ltb/testing/*
                )

        add_library(ltb_gvs_testing ${LTB_SOURCE_FILES})
        ltb_set_properties(ltb_gvs_testing 17)
        target_link_libraries(ltb_gvs_testing PUBLIC LtbExternal::Graphics)
        add_library(LtbGvs::Testing ALIAS ltb_gvs_testing)
    endif ()

    #############################
    ### GVS Display Resources ###
    #############################
    # Compile resources separately to avoid linting issues on generated code
    corrade_add_resource(ltb_gvs_display_RESOURCES ${CMAKE_CURRENT_LIST_DIR}/src/resources.conf)
    add_library(ltb_gvs_display_resources ${ltb_gvs_display_RESOURCES})
    target_link_libraries(ltb_gvs_display_resources
            PUBLIC
            LtbExternal::Graphics
            )
    add_library(LtbGvs::DisplayResources ALIAS ltb_gvs_display_resources)

    ###################
    ### GVS Display ###
    ###################
    file(GLOB_RECURSE LTB_SOURCE_FILES
            LIST_DIRECTORIES false
            CONFIGURE_DEPENDS
            ${CMAKE_CURRENT_LIST_DIR}/src/ltb/gvs/display/*
            )

    ltb_add_library(ltb_gvs_display 17 ${LTB_SOURCE_FILES})
    ltb_link_libraries(ltb_gvs_display
            PUBLIC
            LtbExternal::Gui
            LtbGvs::DisplayResources
            Ltb::Util
            LtbGvs::Core
            )

    target_link_libraries(ltb_gvs INTERFACE ltb_gvs_display)

    #####################
    ### OptiX Backend ###
    #####################
    include(cmake/ThirdPartyCuda.cmake)

    configure_file(
            ${CMAKE_CURRENT_LIST_DIR}/src/gvs_cuda_includes.hpp.in
            ${CMAKE_BINARY_DIR}/generated/ltb/gvs_cuda_includes.hpp
    )

    if (CUDAToolkit_FOUND)
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

        ltb_add_library(ltb_gvs_cuda 17 ${LTB_CUDA_SOURCE_FILES}
                $<$<BOOL:OptiX_INCLUDE>:${LTB_OPTIX_SOURCE_FILES}>
                )

        ltb_link_libraries(ltb_gvs_cuda
                PUBLIC
                Ltb::Util
                LtbGvs::Core
                LtbExternal::Graphics
                LtbExternal::Cuda
                $<$<BOOL:LTB_BUILD_TESTS>:ltb_gvs_testing>
                LtbExternal::Doctest
                )

        target_link_libraries(ltb_gvs INTERFACE ltb_gvs_cuda)
    endif ()

endif (LTB_ENABLE_GVS_DISPLAY)

if (LTB_BUILD_GVS_EXAMPLES)
    add_subdirectory(examples)
endif ()
