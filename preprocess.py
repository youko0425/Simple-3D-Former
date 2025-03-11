import numpy as np
from PIL import Image
import torch
from transformers import ViTFeatureExtractor
import matplotlib.pyplot as plt

# 讀取 .off 文件
def read_off(file_path):
    with open(file_path, 'r') as file:
        header = file.readline().strip()
        if not header.startswith('OFF'):
            raise Exception(f'Not a valid OFF file: {header}')
        
        if len(header) > 3:
            n_verts, n_faces, _ = tuple(map(int, header[3:].split()))
        else:
            n_verts, n_faces, _ = tuple(map(int, file.readline().strip().split(' ')))
        
        verts = []
        for _ in range(n_verts):
            verts.append(tuple(map(float, file.readline().strip().split(' '))))
        
        faces = []
        for _ in range(n_faces):
            faces.append(tuple(map(int, file.readline().strip().split(' '))))
        
    return np.array(verts), np.array(faces)

# 將點雲轉換為多視角圖像
def point_cloud_to_multi_view_images(point_cloud, bins=128):
    top_view = np.histogram2d(point_cloud[:,0], point_cloud[:,1], bins=bins)[0]
    front_view = np.histogram2d(point_cloud[:,0], point_cloud[:,2], bins=bins)[0]
    side_view = np.histogram2d(point_cloud[:,1], point_cloud[:,2], bins=bins)[0]
    
    multi_view_image = np.stack([top_view, front_view, side_view], axis=-1)

    multi_view_image = (multi_view_image - multi_view_image.min()) / (multi_view_image.max() - multi_view_image.min())
    
    return multi_view_image

# 讀取文件並處理
file_path = r'C:\Users\sensh\Desktop\ModelNet40\cup\train\cup_0056.off' ##change if needed
vertices, faces = read_off(file_path)
multi_view_image = point_cloud_to_multi_view_images(vertices, bins=128)

# 顯示多視角圖像


plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(multi_view_image[:,:,0], cmap='viridis')
plt.title("Top View")

plt.subplot(1, 3, 2)
plt.imshow(multi_view_image[:,:,1], cmap='viridis')
plt.title("Front View")

plt.subplot(1, 3, 3)
plt.imshow(multi_view_image[:,:,2], cmap='viridis')
plt.title("Side View")
plt.show()
