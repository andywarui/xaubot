cmake_minimum_required(VERSION 3.15)
project(lightgbm_mt5_dll)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Find LightGBM - Try multiple methods
if(DEFINED ENV{VCPKG_ROOT})
    set(VCPKG_ROOT $ENV{VCPKG_ROOT})
elseif(EXISTS "C:/vcpkg")
    set(VCPKG_ROOT "C:/vcpkg")
endif()

# Method 1: Try to find package
find_package(LightGBM CONFIG QUIET)

if(NOT LightGBM_FOUND)
    # Method 2: Manual include/lib paths from vcpkg
    if(VCPKG_ROOT)
        message(STATUS "Using vcpkg at: ${VCPKG_ROOT}")
        set(LIGHTGBM_INCLUDE_DIR "${VCPKG_ROOT}/installed/x64-windows/include")
        set(LIGHTGBM_LIB_DIR "${VCPKG_ROOT}/installed/x64-windows/lib")

        if(EXISTS "${LIGHTGBM_INCLUDE_DIR}/LightGBM")
            message(STATUS "Found LightGBM headers: ${LIGHTGBM_INCLUDE_DIR}")
            include_directories(${LIGHTGBM_INCLUDE_DIR})
            link_directories(${LIGHTGBM_LIB_DIR})
            set(LightGBM_FOUND TRUE)
        else()
            message(FATAL_ERROR "LightGBM not found in vcpkg. Run: vcpkg install lightgbm:x64-windows")
        endif()
    else()
        message(FATAL_ERROR "vcpkg not found. Install vcpkg first.")
    endif()
endif()

# Create DLL
add_library(lightgbm_mt5 SHARED
    lightgbm_wrapper.cpp
)

# Link LightGBM
if(TARGET LightGBM::LightGBM)
    target_link_libraries(lightgbm_mt5 PRIVATE LightGBM::LightGBM)
else()
    target_link_libraries(lightgbm_mt5 PRIVATE lib_lightgbm)
endif()

# Set DLL properties
set_target_properties(lightgbm_mt5 PROPERTIES
    PREFIX ""
    OUTPUT_NAME "lightgbm_mt5"
    SUFFIX ".dll"
)

# Copy to MT5 Libraries folder after build
add_custom_command(TARGET lightgbm_mt5 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    $<TARGET_FILE:lightgbm_mt5>
    "${CMAKE_SOURCE_DIR}/../MT5_XAUBOT/Libraries/lightgbm_mt5.dll"
    COMMENT "Copying DLL to MT5_XAUBOT/Libraries/"
)
