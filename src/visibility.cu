#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>


namespace py = pybind11;


__global__ void initialize_output_kernel(float* d_output, const int img_h, const int img_w, float init_num) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < img_w && y < img_h) {
        int pixel_index = (y * img_w + x) * 3;
        d_output[pixel_index + 0] = static_cast<float>(x);  // x座標を初期化
        d_output[pixel_index + 1] = static_cast<float>(y);  // y座標を初期化
        d_output[pixel_index + 2] = init_num;               // z座標を初期化（任意の初期値）
    }
}


__device__ float atomicMinFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed, __float_as_int(fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__ void extract_min_z_points_kernel(const float* d_input, const int img_h, const int img_w, const size_t num_points, float* d_output) { 
    const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // スレッドが最小z値を更新
    if (global_idx < num_points) {
        const int x = static_cast<int>(d_input[global_idx * 3 + 0]);  // x座標をグローバルメモリから取得
        const int y = static_cast<int>(d_input[global_idx * 3 + 1]);  // y座標をグローバルメモリから取得
        const float z = d_input[global_idx * 3 + 2];  // z座標を取得

        if (x >= 0 && x < img_w && y >= 0 && y < img_h) {
            atomicMinFloat(&d_output[(y * img_w + x) * 3 + 2], z);  
        }
    }
}


__global__ void occlusion_processing_kernel(float* d_output, const int img_h, const int img_w, const int R, const float threshold, const float init_num) {
    extern __shared__ float shared_data[];

    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

    const int local_x = threadIdx.x + R;  // 共有メモリ内でのx座標（余分なバッファを確保）
    const int local_y = threadIdx.y + R;  // 共有メモリ内でのy座標

    // ブロックの端のスレッドが参照するための余分な境界ピクセルのバッファを確保
    if (x < img_w && y < img_h) {
        shared_data[local_y * (blockDim.x + 2 * R) + local_x] = d_output[(y * img_w + x) * 3 + 2];  // depthのみを保存
    } else {
        shared_data[local_y * (blockDim.x + 2 * R) + local_x] = init_num;
    }

    // 境界ピクセルを共有メモリにロード
    // 左の境界
    if (threadIdx.x == 0 && blockIdx.x > 0)
    { 
        const int left_y = y;
        
        for (int i = 1; i <= R; ++i)
        {
            const int left_x = blockDim.x * (blockIdx.x - 1) + blockDim.x - i;
            
            shared_data[(local_y) * (blockDim.x + 2 * R) + (local_x - i)] = d_output[(left_y * img_w + left_x) * 3 + 2];
        }
    }

    // 右の境界
    if (threadIdx.x == blockDim.x - 1 && blockIdx.x < gridDim.x - 1)
    {
        const int right_y = y;

        for (int i = 1; i <= R; ++i)
        {
            const int right_x = blockDim.x * (blockIdx.x + 1) + i - 1;
            
            shared_data[(local_y) * (blockDim.x + 2 * R) + (local_x + i)] = d_output[(right_y * img_w + right_x) * 3 + 2];
        }
    }

    // 上の境界
    if (threadIdx.y == 0 && blockIdx.y > 0)
    {
        const int top_x = x;

        for (int i = 1; i <= R; ++i)
        {
            const int top_y = blockDim.y * (blockIdx.y - 1) + blockDim.y - i;
            
            shared_data[(local_y - i) * (blockDim.x + 2 * R) + (local_x)] = d_output[(top_y * img_w + top_x) * 3 + 2];
        }
    }

    // 下の境界
    if (threadIdx.y == blockDim.y - 1 && blockIdx.y < gridDim.y - 1)
    {
        const int bottom_x = x;

        for (int i = 1; i <= R; ++i)
        {
            const int bottom_y = blockDim.y * (blockIdx.y + 1) + i - 1;
            
            shared_data[(local_y + i) * (blockDim.x + 2 * R) + (local_x)] = d_output[(bottom_y * img_w + bottom_x) * 3 + 2];
        }
    }

    // 左上の境界
    if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x > 0 && blockIdx.y > 0)
    {         
        for (int j = 1; j <= R; ++j)
        {
            const int upper_left_y = blockDim.y * (blockIdx.y - 1) + blockDim.y - j;
            
            for (int i = 1; i <= R; ++i)
            {
                const int upper_left_x = blockDim.x * (blockIdx.x - 1) + blockDim.x - i;
            
                shared_data[(local_y - j) * (blockDim.x + 2 * R) + (local_x - i)] = d_output[(upper_left_y * img_w + upper_left_x) * 3 + 2];
            }
        }
    }

    // 右上の境界
    if (threadIdx.x == blockDim.x - 1 && threadIdx.y == 0 && blockIdx.x < gridDim.x - 1 && blockIdx.y > 0)
    {         
        for (int j = 1; j <= R; ++j)
        {
            const int upper_right_y = blockDim.y * (blockIdx.y - 1) + blockDim.y - j;
            
            for (int i = 1; i <= R; ++i)
            {
                const int upper_right_x = blockDim.x * (blockIdx.x + 1) + i - 1;
            
                shared_data[(local_y - j) * (blockDim.x + 2 * R) + (local_x + i)] = d_output[(upper_right_y * img_w + upper_right_x) * 3 + 2];
            }
        }
    }

    // 左下の境界
    if (threadIdx.x == 0 && threadIdx.y == blockDim.y - 1 && blockIdx.x > 0 && blockIdx.y < gridDim.y - 1)
    {         
        for (int j = 1; j <= R; ++j)
        {
            const int lower_left_y = blockDim.y * (blockIdx.y + 1) +  j - 1;
            
            for (int i = 0; i <= R; ++i)
            {
                const int lower_left_x = blockDim.x * (blockIdx.x - 1) + blockDim.x - i;
            
                shared_data[(local_y + j) * (blockDim.x + 2 * R) + (local_x - i)] = d_output[(lower_left_y * img_w + lower_left_x) * 3 + 2];
            }
        }
    }

    // 右下の境界
    if (threadIdx.x == blockDim.x - 1 && threadIdx.y == blockDim.y - 1 && blockIdx.x < gridDim.x - 1 && blockIdx.y < gridDim.y - 1)
    {         
        for (int j = 1; j <= R; ++j)
        {
            const int lower_right_y = blockDim.y * (blockIdx.y + 1) + j - 1;
            
            for (int i = 0; i <= R; ++i)
            {
                const int lower_right_x = blockDim.x * (blockIdx.x + 1) + i - 1;
            
                shared_data[(local_y + j) * (blockDim.x + 2 * R) + (local_x + i)] = d_output[(lower_right_y * img_w + lower_right_x) * 3 + 2];
            }
        }
    }
    

    __syncthreads();  // 全てのスレッドが共有メモリにデータをロードするのを待機

    if (x >= img_w || y >= img_h) return;

    const float pixel = shared_data[local_y * (blockDim.x + 2 * R) + local_x];

    int sum = 0;
    int count = 0;

    if (pixel >= init_num) return;
    

    for (int dx = -R; dx <= R; ++dx) {
        for (int dy = -R; dy <= R; ++dy) {
            if (dx == 0 && dy == 0) continue;

            const int nx = local_x + dx;
            const int ny = local_y + dy;
            const int temp_x = x + dx;
            const int temp_y = y + dy;

            if (temp_x < 0 || temp_x >= img_w || temp_y < 0 || temp_y >= img_h) continue; 
            
            const float temp_pixel = shared_data[(ny * (blockDim.x + 2 * R) + nx)];

            if (temp_pixel < init_num) {
                count++;
                if (temp_pixel < pixel - 3.0f)
                    sum++;
            }
            
        }
    }

    if (sum >= 1 + (threshold * count) / (R * R * 2 * 2)) d_output[(y * img_w + x) * 3 + 2] = init_num;

}


class OcclusionProcessor {
public:
    OcclusionProcessor(const py::object input_cupy, const int img_h, const int img_w)
        : img_h(img_h), img_w(img_w), d_output(nullptr) {

        // CuPy配列からGPUメモリポインタを直接取得
        const auto ptr = input_cupy.attr("data").attr("ptr").cast<std::uintptr_t>();
        input_ptr = reinterpret_cast<float*>(ptr);

        // 配列の形状を取得
        py::tuple shape = input_cupy.attr("shape").cast<py::tuple>();
        const int num_points = shape[0].cast<size_t>();
        
        // 入力がfloat32であることを確認
        if (input_cupy.attr("dtype").attr("name").cast<std::string>() != "float32") {
            std::cout << "dtype is " << input_cupy.attr("dtype").attr("name").cast<std::string>() << std::endl;
            throw std::runtime_error("Input must be float32");
        }

        // メモリ確保のエラーチェック
        cudaError_t err = cudaMalloc(&d_output, sizeof(float) * img_h * img_w * 3);
        if (err != cudaSuccess) {
            std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
            throw std::runtime_error("CUDA memory allocation failed");
        }

        launch_initialize_output();
        launch_extract_min_z_points(num_points);
        launch_occlusion_processing();
    }

    // 結果を返すメソッド
    py::tuple get_result() const {
        return py::make_tuple(reinterpret_cast<std::uintptr_t>(d_output), img_h * img_w , 3, init_num);
    }

    // CUDAメモリの解放
    void free_memory() {
        if (d_output != nullptr) {
            cudaError_t err = cudaFree(static_cast<void*>(d_output));  // CUDAメモリの解放
            if (err != cudaSuccess)
            {
                std::cerr << "cudaFree failed: " << cudaGetErrorString(err) << std::endl;
            }
            
            d_output = nullptr;  // ポインタを無効にする
            std::cout << "CUDA memory freed." << std::endl;
        }
    }

    // メモリ量を計算して表示するメソッド
    void display_memory_usage() const {
        size_t free_mem = 0;
        size_t total_mem = 0;

        cudaError_t err = cudaMemGetInfo(&free_mem, &total_mem);
        if (err != cudaSuccess) {
            std::cerr << "Error getting memory info: " << cudaGetErrorString(err) << std::endl;
            return; // エラー発生時は処理を中断
        }

        std::cout << "Free GPU Memory: " << free_mem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "Total GPU Memory: " << total_mem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "Requested Memory: " << sizeof(float) * img_h * img_w * 3 / (1024 * 1024) << " MB" << std::endl;
    }

    void display_input_array(const int display_points) const {
        const int size = display_points * 3;  // 配列のサイズ
        std::vector<float> h_array(size);      // ホスト側メモリを確保

        // GPUメモリからホストメモリにコピー
        cudaError_t err = cudaMemcpy(h_array.data(), input_ptr, size * sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(err) << std::endl;
            return; // エラー発生時は処理を中断
        }

        // ポインタアドレス順に表示
        for (int i = 0; i < size; ++i) {
            std::cout << "Address: " << &h_array[i] << " Value: " << h_array[i] << std::endl;
        }
    }

private:
    float* d_output;  
    const float* input_ptr; 
    const int img_h;
    const int img_w;
    const int R = 3;
    const float threshold = 3.0f;
    const float init_num = 1000;

    void launch_initialize_output();
    void launch_extract_min_z_points(const size_t num_points);
    void launch_occlusion_processing();
};

void OcclusionProcessor::launch_initialize_output() {
    const dim3 blockSize(16, 16); // 各ブロックのスレッド数
    const dim3 gridSize((img_w + blockSize.x - 1) / blockSize.x, (img_h + blockSize.y - 1) / blockSize.y); // グリッドのサイズ

    // 出力配列を (x, y, init_num) で初期化するカーネルの呼び出し
    initialize_output_kernel<<<gridSize, blockSize>>>(d_output, img_h, img_w, init_num);
    cudaDeviceSynchronize(); // エラーチェックのために同期を取る

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }
}

void OcclusionProcessor::launch_extract_min_z_points(const size_t num_points) {
    const dim3 blockSize(256);  // 各ブロックのスレッド数
    const dim3 gridSize((num_points + blockSize.x - 1) / blockSize.x);  // グリッドのサイズ

    extract_min_z_points_kernel<<<gridSize, blockSize>>>(input_ptr, img_h, img_w, num_points, d_output);
    cudaDeviceSynchronize();  // エラーチェックのために同期を取る

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }
}

void OcclusionProcessor::launch_occlusion_processing() {
    const dim3 blockSize(16, 16); // 各ブロックのスレッド数
    const dim3 gridSize((img_w + blockSize.x - 1) / blockSize.x, (img_h + blockSize.y - 1) / blockSize.y); // グリッドのサイズ
    const size_t shared_mem_size = (blockSize.x + 2 * R) * (blockSize.y + 2 * R) * sizeof(float);  // 共有メモリサイズ

    occlusion_processing_kernel<<<gridSize, blockSize, shared_mem_size>>>(d_output, img_h, img_w, R, threshold, init_num);
    cudaDeviceSynchronize();  // エラーチェックのために同期を取る

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }
}

PYBIND11_MODULE(visibility, m) {
    py::class_<OcclusionProcessor>(m, "OcclusionProcessor")
        .def(py::init<const py::object, const int, const int>(), py::arg("input_cupy"), py::arg("img_h"), py::arg("img_w"))
        .def("get_result", &OcclusionProcessor::get_result, "Get the processed result")
        .def("free_memory", &OcclusionProcessor::free_memory, "Free the allocated CUDA memory")
        .def("display_memory_usage", &OcclusionProcessor::display_memory_usage, "Display GPU memory usage")
        .def("display_input_array", &OcclusionProcessor::display_input_array, "Display input array content");
}