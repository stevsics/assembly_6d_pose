"""Main script for plotting objects using dataset information."""

import os
import cv2
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

from utils.read_obj_file import MeshData
from utils.plot_utils import overlay_mesh_wireframe_2_img
from bounding_box_3d.cuboid import Cuboid


def plot_target_pose(dataset_folder, mesh_file, camera_params):

    # read mesh
    object_model = MeshData(mesh_file)

    mesh_params = {
        "vertices": np.array(object_model.get_vertices()),
        "faces": object_model.get_faces(),
    }

    it_sample = 0
    file_exists = True

    while file_exists:

        # get input image
        image_file_name = os.path.join(dataset_folder, "%06d" % it_sample + ".png")
        if not os.path.exists(image_file_name):
            file_exists = False
            continue

        img = cv2.imread(image_file_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # get position and rotation matrix
        json_file = os.path.join(dataset_folder, "%06d" % it_sample + ".json")
        with open(json_file) as f:
            json_data = json.load(f)
        t_vec = np.array(json_data["target"]["translation"])
        R_mat = np.array(json_data["target"]["rotation_matrix"])

        # overlay wireframe of the object mesh to the image
        img = overlay_mesh_wireframe_2_img(img, R_mat, t_vec, mesh_params, camera_params, color=(255, 0, 255), thickness=2)

        plt.figure()
        plt.imshow(img)
        plt.show()

        it_sample += 1


def generate_keypoints_and_plot_3D_bounding_box(dataset_folder, bb_file, camera_params):

    # read cuboid config file
    with open(bb_file) as f:
        json_data = json.load(f)
    bb_cuboid = Cuboid(np.array([json_data["width"], json_data["depth"], json_data["height"]]))

    it_sample = 0
    file_exists = True

    while file_exists:

        # get input image
        image_file_name = os.path.join(dataset_folder, "%06d" % it_sample + ".png")
        if not os.path.exists(image_file_name):
            file_exists = False
            continue

        img = cv2.imread(image_file_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # get position and rotation matrix
        json_file = os.path.join(dataset_folder, "%06d" % it_sample + ".json")
        with open(json_file) as f:
            json_data = json.load(f)
        t_vec = np.array(json_data["target"]["translation"])
        R_mat = np.array(json_data["target"]["rotation_matrix"])

        # overlay wireframe of the bounding box to the image
        plot_img = bb_cuboid.plot(img.copy(), R_mat, t_vec,
                                  camera_params["intrinsic_matrix"],
                                  camera_params["dist_coeffs"])
        plt.figure()
        plt.imshow(plot_img)

        # get keypoints from the corners of the bounding box
        keypoints = bb_cuboid.get_2d_projection(R_mat, t_vec,
                                                camera_params["intrinsic_matrix"],
                                                camera_params["dist_coeffs"])

        # plot keypoints over the image
        for it_keyoint in keypoints:
            cv2.circle(img, tuple(it_keyoint.astype(dtype=np.int32)), 1, (255, 0, 255), thickness=2)

        plt.figure()
        plt.imshow(img)
        plt.show()

        it_sample += 1


def plot_from_arguments(dataset_folder, task_name, dataset, mesh_name='manipulation_object', plot_mesh='mesh'):

    # choose the dataset for plotting
    dataset_folder_task = os.path.join(dataset_folder, task_name)
    if dataset == "train_original_mesh":
        dataset_folder_selection = os.path.join(dataset_folder_task, "train_data", "original_mesh")
    elif dataset == "train_random_mesh":
        dataset_folder_selection = os.path.join(dataset_folder_task, "train_data", "random_mesh")
    elif dataset == "test":
        dataset_folder_selection = os.path.join(dataset_folder_task, "test_data", "object_1")
    else:
        print("Invalid dataset type!")
        exit()

    json_file = os.path.join(dataset_folder, "config_files", "camera_parameters.json")
    with open(json_file) as f:
        json_data = json.load(f)

    camera_params = {
        "intrinsic_matrix": np.array(json_data["intrinsic_matrix"]),
        "dist_coeffs": np.array(json_data["dist_coeffs"])
    }

    # choose to plot mesh or bounding box
    if plot_mesh == 'mesh':
        mesh_file = os.path.join(dataset_folder_task, "models", mesh_name+".obj")
        plot_target_pose(dataset_folder_selection, mesh_file, camera_params)
    elif plot_mesh == 'bb':
        bb_file = os.path.join(dataset_folder_task, "models", "bounding_box.json")
        generate_keypoints_and_plot_3D_bounding_box(dataset_folder_selection, bb_file, camera_params)
    else:
        print("Invalid plot_mesh type!")
        exit()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, default='TODO:DATASET_PATH', help='Path to the folder where the dataset is located.')
    parser.add_argument('--task_name', type=str, default='task_1', help='Task name. Options: task_1, task_2, task_3, task_4.')
    parser.add_argument('--dataset', type=str, default='test', help='The dataset variable is used to choose a dataset type.'
                                                                'Options: train_original_mesh, train_random_mesh, test.')
    parser.add_argument('--mesh_name', type=str, default='manipulation_object', help='Name of the mesh to plot.'
                                                                                     'Options: manipulation_object, target_object, template_mesh.')
    parser.add_argument('--plot_mesh', type=str, default='mesh', help='Plot the object mesh or bounding box. Options: mesh, bb.')
    opt = parser.parse_args()

    plot_from_arguments(opt.dataset_root, opt.task_name, opt.dataset, mesh_name=opt.mesh_name, plot_mesh=opt.plot_mesh)
