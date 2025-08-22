cmake_minimum_required(VERSION 3.22)

project(CG)
include(FetchContent)

FetchContent_Declare(
  Kokkos
  GIT_REPOSITORY https://github.com/kokkos/kokkos.git
  GIT_TAG        4.6.02
)

FetchContent_Declare(
  KokkosKernels
  GIT_REPOSITORY https://github.com/kokkos/kokkos-kernels.git
  GIT_TAG        4.6.02
  CMAKE_ARGS     -DCMAKE_CXX_FLAGS="-Wno-deprecated-declarations"
)

message(STATUS "kokkos...")
FetchContent_MakeAvailable(Kokkos)

message(STATUS "kokkos-kernels...")
FetchContent_MakeAvailable(KokkosKernels)


add_executable(main main.cpp)
target_link_libraries(main Kokkos::kokkos Kokkos::kokkoskernels)

enable_testing()
