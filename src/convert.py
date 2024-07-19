import numpy as np
from scipy.io import savemat

def read_off(file_path):
    with open(file_path, 'r') as file:
        if 'OFF' != file.readline().strip():
            raise ValueError('Not a valid OFF header')
        n_verts, n_faces, _ = tuple(map(int, file.readline().strip().split()))
        vertices = [tuple(map(float, file.readline().strip().split())) for _ in range(n_verts)]
        faces = [tuple(map(int, file.readline().strip().split()[1:])) for _ in range(n_faces)]
        return np.array(vertices), np.array(faces)

def save_to_mat(vertices, faces, mat_file_path):
    mdic = {'vertices': vertices, 'faces': faces}
    savemat(mat_file_path, mdic)

# Example usage:
off_file_path = '../data/archive/ModelNet10/chair/train/chair_0001.off'
mat_file_path = 'output_mat_file.mat'

vertices, faces = read_off(off_file_path)
save_to_mat(vertices, faces, mat_file_path)
