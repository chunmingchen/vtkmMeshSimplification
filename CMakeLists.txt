cmake_minimum_required(VERSION 2.8.11)

project(VTKMMeshSimplify CXX)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${mccompare_SOURCE_DIR})

set(DEVICE "Serial" CACHE STRING "vtkm_configure_device")

#########################################################
# Include VTK
#########################################################
find_package(VTKm REQUIRED)
include(VTKmMacros)
vtkm_configure_device(${DEVICE})

string(TOUPPER ${DEVICE} DEVICE_UPPER)
message(STATUS "VTKM_DEVICE_ADAPTER_${DEVICE_UPPER}")
add_definitions(-DVTKM_DEVICE_ADAPTER=VTKM_DEVICE_ADAPTER_${DEVICE_UPPER})

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

########################################################
# Include TBB
########################################################
find_package(TBB)
include_directories(${TBB_INCLUDE_DIRS})
link_directories(${TBB_LIBRARY_DIRS})



########################################################
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include_directories(.)

#########################################################
# Data path
#########################################################
set(DATA_FILE "/home/jimmychen/Project/VTKData/Data/cow.vtp" CACHE FILEPATH "Data filename")
add_definitions(-DDATA_FILE=\"${DATA_FILE}\")

#########################################################
set(headers  
  simplify.h
  algorithm.h
  math/Matrix.h
  math/VectorAnalysis.h
  math/Basic.h
)

set(srcs
  simplify.cxx
  )

set_source_files_properties(${headers} PROPERTIES HEADER_FILE_ONLY TRUE)

add_executable(Benchmark
  ${srcs}
  ${headers}
  main.cxx
  )

target_link_libraries(Benchmark ${VTK_LIBRARIES} ${TBB_LIBRARIES} )

#Add CUDA version
if (${DEVICE}=="CUDA") # string compare?
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
endif()

if(VTK_HAS_FLYING_EDGES)
  target_compile_definitions(Benchmark PRIVATE "VTK_HAS_FLYING_EDGES")
endif()
