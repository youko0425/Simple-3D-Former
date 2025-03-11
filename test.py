import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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
            face_data = list(map(int, file.readline().strip().split(' ')))
            faces.append(face_data[1:])  # Skip the first number which indicates the number of vertices per face
        
    return np.array(verts), faces

def visualize_off_files(file_paths):
    for file_path in file_paths:
        # Reading 3D point cloud and face
        vertices, faces = read_off(file_path)
        print(f"Vertices: {vertices.shape}, Faces: {len(faces)}")
        
        # Visualizing 3D point cloud
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot for vertices
        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=1)

        # Plotting the faces
        face_vertices = [[vertices[idx] for idx in face] for face in faces]
        poly3d = Poly3DCollection(face_vertices, alpha=0.3, facecolor='cyan', edgecolor='k')
        ax.add_collection3d(poly3d)

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        plt.title(f'Visualization of {file_path}')
        plt.show()

# List of file paths
# ##change if needed
file_paths = [
    r'C:\Users\sensh\Desktop\ModelNet40\airplane\train\airplane_0062.off',
    r'C:\Users\sensh\Desktop\ModelNet40\airplane\train\airplane_0075.off',
    r'C:\Users\sensh\Desktop\ModelNet40\airplane\train\airplane_0001.off',
    r'C:\Users\sensh\Desktop\ModelNet40\airplane\train\airplane_0100.off',
    r'C:\Users\sensh\Desktop\ModelNet40\airplane\train\airplane_0087.off',
]

# Visualizing all OFF files
visualize_off_files(file_paths)