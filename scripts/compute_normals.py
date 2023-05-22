import pymeshlab
import sys, os, shutil
from os import walk
import numpy as np
from numpy import linalg as lg

ms = pymeshlab.MeshSet()


file = sys.argv[1]
operation = int(sys.argv[2])
param = float(sys.argv[3])
filename = file[:-4]

if (not os.path.isfile(filename+'_with_normal.xyz')) and ("with_normal" not in filename):
    print("adding normals for ", filename)
    ms.load_new_mesh(file)
    ms.compute_normal_for_point_clouds()
    ms.save_current_mesh(filename+'_with_normal.xyz')


def add_gaussian_noise(raw_pts, sigma):
    new_pts = raw_pts + np.random.normal(0, sigma, (raw_pts.shape[0], raw_pts.shape[1]))
    return new_pts

def add_Gaussian_noise_bbox(raw_pts, sigma):
    bbox = np.amax(raw_pts, axis=0)-np.amin(raw_pts, axis=0)
    diag_length = lg.norm(bbox)
    print("diag_length = ", diag_length)
    print("actual sigma = ", diag_length * sigma)
    return add_gaussian_noise(raw_pts, diag_length*sigma)

    
if operation == 1:
    # gaussian noise
    if not os.path.isfile(filename+'_random_gaussian_normal.xyz'):
        raw_pts = np.genfromtxt(fname=filename+'_with_normal.xyz')
        normals = raw_pts[:, -3:]
        p_normals = add_Gaussian_noise_bbox(normals, param)
        raw_pts[:, -3:] = p_normals
        np.savetxt(filename+'_random_gaussian_normal.xyz', raw_pts, delimiter=' ')

    if not os.path.isfile(filename+'_random.xyz'):
        raw_pts = np.genfromtxt(fname=filename+'.xyz')
        random_normals = np.random.rand(len(raw_pts), 3)
        new_pts = np.concatenate((raw_pts, random_normals), axis=1)
        np.savetxt(filename+'_random.xyz', new_pts, delimiter=' ')



    if not os.path.isfile(filename+'_one.xyz'):
        raw_pts = np.genfromtxt(fname=filename+'.xyz')
        one_normals = np.ones((len(raw_pts), 3))
        new_pts = np.concatenate((raw_pts, one_normals), axis=1)
        np.savetxt(filename+'_one.xyz', new_pts, delimiter=' ')
