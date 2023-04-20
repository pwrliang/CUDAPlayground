cmake_minimum_required(VERSION 3.16)
project(CUDAPlayground LANGUAGES C CXX CUDA)

# ------------------------------------------------------------------------------
# setting default cmake type to Release
# ------------------------------------------------------------------------------

set(DEFAULT_BUILD_TYPE "Release")
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE
            STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
            "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif ()

find_package(CUDAToolkit)

set(CMAKE_CUDA_ARCHITECTURES "60;61;70")

############## BUILD ##############

include_directories(${PROJECT_SOURCE_DIR}/include)
set(EXTRA_LIBS pthread CUDA::nvToolsExt CUDA::cuda_driver)
set(CMAKE_CUDA_STANDARD 14)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    message("Debug mode")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -lineinfo -Xcompiler -ggdb -std=c++14 --extended-lambda")
else ()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -DNDEBUG -Xcompiler -DNDEBUG -std=c++14 --extended-lambda")
endif ()


include_directories(${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/src)

add_executable(main src/main.cu)
target_link_libraries(main ${EXTRA_LIBS})

