"""This class plots a bounding box and generates the keypoints."""

import numpy as np
import cv2

class Cuboid():

    # All adges in cuboid. Same for every cuboid
    edges = np.array([
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ])

    # Create a cuboid with a given size
    def __init__(self, size3d = [1.0, 1.0, 1.0], center_location = [0.0, 0.0, 0.0]):

        width, depth, height = size3d
        cx, cy, cz = center_location

        right = cx + width / 2.0
        left = cx - width / 2.0
        front = cy + depth / 2.0
        rear = cy - depth / 2.0
        top = cz + height / 2.0
        bottom = cz - height / 2.0

        # List of 8 vertices of the cuboid
        self.vertices = np.array([
            [right, top, front],  # Front Top Right
            [left, top, front],  # Front Top Left
            [left, bottom, front],  # Front Bottom Left
            [right, bottom, front],  # Front Bottom Right
            [right, top, rear],  # Rear Top Right
            [left, top, rear],  # Rear Top Left
            [left, bottom, rear],  # Rear Bottom Left
            [right, bottom, rear],  # Rear Bottom Right
        ])

    def plot(self, img, R_mat, t_vec, camera_intrinsics, dist_coeffs, config=None):
        if config is None:
            config = {
                "color": (255, 0, 255),
                "thickness": 2,
            }

        projected_points = self.get_2d_projection(R_mat, t_vec, camera_intrinsics, dist_coeffs)

        for it_edges in Cuboid.edges:
            cv2.line(img, tuple(projected_points[it_edges[0], :].astype(dtype=np.int32)),
                     tuple(projected_points[it_edges[1], :].astype(dtype=np.int32)),
                     color=config["color"], thickness=config["thickness"])

        return img

    def get_2d_projection(self, R_mat, t_vec, camera_intrinsics, dist_coeffs):

        projected_points, _ = cv2.projectPoints(self.vertices, cv2.Rodrigues(R_mat)[0], t_vec,
                                                camera_intrinsics,
                                                dist_coeffs)

        return np.squeeze(projected_points)