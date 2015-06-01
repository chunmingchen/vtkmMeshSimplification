cmake_minimum_required(VERSION 2.8.11)

project(VTKMMeshSimplify CXX)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${mccompare_SOURCE_DIR})

option(USE_CUDA "Compile with cuda" OFF)

#########################################################
# Include VTKm
#########################################################
find_package(VTKm REQUIRED)
include(VTKmMacros)
vtkm_configure_device(Serial)
#vtkm_configure_device(Cuda)


#########################################################
# Include VTK
#########################################################
find_package(VTK 6.2 REQUIRED)
include(${VTK_USE_FILE})


########################################################
# Include Boost
########################################################
find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
########################################################


#########################################################
# Data path
#########################################################
set(DATA_FILE "/home/jimmychen/Project/VTKData/Data/cow.vtp" CACHE FILEPATH "Data filename")
add_definitions(-DDATA_FILE=\"${DATA_FILE}\")

#########################################################
set(headers  
  simplify.h
)

set(srcs
  simplify.cxx
  )

set_source_files_properties(${headers} PROPERTIES HEADER_FILE_ONLY TRUE)

add_executable(BenchmarkSerial
  ${srcs}
  ${headers}
  mainSerial.cxx
  )

target_link_libraries(BenchmarkSerial ${VTK_LIBRARIES}  )

#Add CUDA version
if (USE_CUDA)
    cuda_add_executable(BenchmarkCuda
      ${srcs}
      ${headers}
      mainCuda.cu
      )

    target_link_libraries(BenchmarkCuda
      vtkCommonCore
      vtkCommonDataModel
      vtkCommonExecutionModel
      vtkCommonMisc
      vtkFiltersCore
      vtkFiltersGeometry
      vtkImagingCore
      vtkIOImage
      vtkIOLegacy
      )
endif(USE_CUDA)

if(VTK_HAS_FLYING_EDGES)
  target_compile_definitions(BenchmarkSerial PRIVATE "VTK_HAS_FLYING_EDGES")
  #target_compile_definitions(BenchmarkCuda   PRIVATE "VTK_HAS_FLYING_EDGES")
endif()
