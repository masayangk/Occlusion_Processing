cmake_minimum_required(VERSION 3.10)
project(Occlusion_Project LANGUAGES CXX CUDA C)

# C++17を指定
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDAアーキテクチャを指定
set(CMAKE_CUDA_ARCHITECTURES "61;70;75;80;86")

# pybind11の設定
find_package(pybind11 REQUIRED)

# CUDAを見つける
find_package(CUDA REQUIRED)

# ヘッダーのインクルードパスを追加
include_directories(${CUDA_INCLUDE_DIRS})

# モジュールを作成
pybind11_add_module(visibility src/visibility.cu)

# リンクライブラリを設定
target_link_libraries(visibility PRIVATE ${CUDA_CUDART_LIBRARY} pybind11::module)

# 最適化オプションの設定
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")