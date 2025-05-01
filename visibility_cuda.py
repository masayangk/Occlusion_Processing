import visibility
import numpy as np
import cupy as cp
from scipy.spatial.transform import Rotation
from plyfile import PlyData
import cv2
import matplotlib.pyplot as plt

png_file_path = f'/workspace/data/sample.png'
ply_file_path = f'/workspace/data/sample.ply'
intrinsic_file_path = f'/workspace/data/intrinsic.txt'

# イベントの作成（開始と終了）
start = cp.cuda.Event()
end = cp.cuda.Event()

# 処理前にGPUの同期を行う
cp.cuda.Stream.null.synchronize()

bgr = cv2.imread(png_file_path, cv2.IMREAD_COLOR)
img_h, img_w = bgr.shape[:2]
ply_data = PlyData.read(ply_file_path)

# x, y, z 座標の取得
x = cp.asarray(ply_data['vertex']['x'])
y = cp.asarray(ply_data['vertex']['y'])
z = cp.asarray(ply_data['vertex']['z'])

with open(intrinsic_file_path, mode='r') as f:
    line = f.readlines()
    fx, fy, cx, cy = map(float, line[0].split(',')[:4])

intrinsics = cp.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=cp.float32)

CMRNet2Img = Rotation.from_euler('xyz', np.array([-90, -90, 0]), degrees=True).as_matrix()
CMRNet2Img = cp.asarray(CMRNet2Img, dtype=cp.float32)  # RotationはSciPyを使うのでNumPyで計算し、CuPyに変換

pcd_input = cp.array([x, y, z], dtype=cp.float32)  # 点群データを (N, 3) の形に変換
pcd_input = cp.dot(CMRNet2Img, pcd_input)
pcd_input = pcd_input[:, pcd_input[2, :] > 0]  # z > 0 のフィルタリング
print(pcd_input.shape)
print(intrinsics.shape)
pcd_input = cp.dot(pcd_input.T, intrinsics.T)  # カメラ座標系から画像座標系への変換
pcd_input[:, :2] /= pcd_input[:, 2][:, cp.newaxis]  # ピクセル座標への正規化
print(f"入力点群: {pcd_input}")

# 計測開始
start.record()

processor = visibility.OcclusionProcessor(pcd_input, img_h, img_w)
d_result_ptr, row, col, init_num = processor.get_result()  # モジュールからCUDAメモリのポインタを取得

# print('d_result_ptr type:', type(d_result_ptr))
# print('d_result_ptr value:', d_result_ptr)
# print('Is pointer aligned?:', d_result_ptr % 8 == 0)  # アラインメントチェック

unowned_mem = cp.cuda.UnownedMemory(
        ptr=d_result_ptr,          # CUDAメモリポインタ
        size=row * col * 4,        # メモリサイズ（float32なので4バイト）
        owner=processor            # メモリの所有者
    )
    
# MemoryPointerとndarrayの作成
mem_ptr = cp.cuda.MemoryPointer(unowned_mem, 0)
pcd_output = cp.ndarray(
    shape=(row, col),
    dtype=cp.float32,
    memptr=mem_ptr
)

# 計測終了
end.record()

# processor.display_memory_usage() # GPUメモリ使用量
# display_points = 5
# processor.display_input_array(display_points) # 任意の数だけアドレス順に表示

# 処理が完了するまで同期を行う
end.synchronize()

# 経過時間をミリ秒単位で取得
elapsed_time = cp.cuda.get_elapsed_time(start, end) / 1000  # 秒に変換

cmap = plt.get_cmap('jet')
pcd_output_cpu: np.ndarray = pcd_output.get()
max = pcd_output_cpu[pcd_output_cpu[:, 2] < init_num][:, 2].max()
pix_colors = cmap(pcd_output_cpu[:, 2] / max)

processor.free_memory() # GPUメモリ開放
# processor.display_memory_usage() # GPUメモリ使用量

for pix, pix_color in zip(pcd_output_cpu, pix_colors):
    if pix[2] >= init_num:
        continue
    
    u, v = pix[:2]
    pix_color = np.asarray((pix_color[:3] * 255)).astype(np.uint8)
    cv2.circle(bgr, (np.uint16(u), np.uint16(v)), 1, tuple(pix_color.tolist()), -1)
    
cv2.imwrite('/workspace/result/output_cu.png', bgr)

# 結果の表示
print(f"処理時間: {elapsed_time} 秒")
print(f"有効点群: {pcd_output_cpu[pcd_output_cpu[:, 2] < init_num].shape}")
print(pcd_output_cpu[pcd_output_cpu[:, 2] < init_num])