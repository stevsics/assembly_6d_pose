"""Utility functions for plotting."""

import numpy as np
import cv2


def overlay_mesh_wireframe_2_img(img, R_mat, t_vec, mesh_params, camera_params, color=(255, 0, 255), thickness=2):

    projected_points, _ = cv2.projectPoints(mesh_params["vertices"], cv2.Rodrigues(R_mat)[0], t_vec, camera_params["intrinsic_matrix"],
                                            camera_params["dist_coeffs"])

    projected_points = np.squeeze(projected_points)


    for it_faces in range(len(mesh_params["faces"])):

        face_it = mesh_params["faces"][it_faces]
        for it_vert in range(len(face_it)):

            point_it_1 = projected_points[face_it[it_vert], :].astype(dtype=np.int32)
            if (it_vert + 1) < len(face_it):
                point_it_2 = projected_points[face_it[it_vert + 1], :].astype(dtype=np.int32)
            else:
                point_it_2 = projected_points[face_it[0], :].astype(dtype=np.int32)

            cv2.line(img, (point_it_1[0], point_it_1[1]), (point_it_2[0], point_it_2[1]),
                     color=color, thickness=thickness)


    return img