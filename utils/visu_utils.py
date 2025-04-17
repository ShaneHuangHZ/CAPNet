import os
import numpy as np
import cv2

def get_3d_bbox(size, shift=0):
    """
    Args:
        size: [3] or scalar
        shift: [3] or scalar
    Returns:
        bbox_3d: [3, N]

    """
    bbox_3d = np.array([[+size[0] / 2, +size[1] / 2, +size[2] / 2],
                        [+size[0] / 2, +size[1] / 2, -size[2] / 2],
                        [-size[0] / 2, +size[1] / 2, +size[2] / 2],
                        [-size[0] / 2, +size[1] / 2, -size[2] / 2],
                        [+size[0] / 2, -size[1] / 2, +size[2] / 2],
                        [+size[0] / 2, -size[1] / 2, -size[2] / 2],
                        [-size[0] / 2, -size[1] / 2, +size[2] / 2],
                        [-size[0] / 2, -size[1] / 2, -size[2] / 2]]) + shift
    bbox_3d = bbox_3d.transpose()
    return bbox_3d
def calculate_2d_projections(coordinates_3d, intrinsics):
    """
    Args:
        coordinates_3d: [3, N]
        intrinsics: [3, 3]

    Returns:
        projected_coordinates: [N, 2]
    """
    projected_coordinates = intrinsics @ coordinates_3d
    projected_coordinates = projected_coordinates[:2, :] / projected_coordinates[2, :]
    projected_coordinates = projected_coordinates.transpose()
    projected_coordinates = np.array(projected_coordinates, dtype=np.int32)

    return projected_coordinates
def transform_coordinates_3d(coordinates, sRT):
    """
    Args:
        coordinates: [3, N]
        sRT: [4, 4]

    Returns:
        new_coordinates: [3, N]

    """
    assert coordinates.shape[0] == 3
    coordinates = np.vstack([coordinates, np.ones((1, coordinates.shape[1]), dtype=np.float32)])
    new_coordinates = sRT @ coordinates
    # if new_coordinates[3, :].any() == 0:
    #     new_coordinates = new_coordinates
    # else:
    #     new_coordinates = new_coordinates[:3, :] / new_coordinates[3, :]
    new_coordinates = new_coordinates[:3, :] / new_coordinates[3, :]
    return new_coordinates

def draw_detections_pcs_pred(img, out_dir, data_name, img_id, intrinsics,
                    gt_sRT, gt_size,draw_gt=True,color=(255,255,0),joint=False):
    """ Visualize pose predictions.
    """
    out_path = os.path.join(out_dir, '{}_{}_pred_pcs_pred.png'.format(data_name, img_id))

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if draw_gt:
        for i in range(gt_sRT.shape[0]):
            sRT = gt_sRT[i, :, :]
            bbox_3d = get_3d_bbox(gt_size[i, :], 0)
            transformed_bbox_3d = transform_coordinates_3d(bbox_3d, sRT)
            projected_bbox = calculate_2d_projections(transformed_bbox_3d, intrinsics)
            xyz_axis = 0.03 * np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]).transpose()
            transformed_axes = transform_coordinates_3d(xyz_axis, sRT)
            projected_axes = calculate_2d_projections(transformed_axes, intrinsics)
            img = draw(img, projected_bbox, projected_axes, color)
    cv2.imwrite(out_path, img)

def draw(img, imgpts, axes, color):
    imgpts = np.int32(imgpts).reshape(-1, 2)
    color_ground = (int(color[0]), int(color[1]), int(color[2]))
    for i, j in zip([4, 5, 6, 7], [5, 7, 4, 6]):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color=color_ground, thickness=2, lineType=cv2.LINE_AA)

    color_pillar = (int(color[0]), int(color[1]), int(color[2]))
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color=color_pillar, thickness=2, lineType=cv2.LINE_AA)

    for i, j in zip([0, 1, 2, 3], [1, 3, 0, 2]):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color=color, thickness=2, lineType=cv2.LINE_AA)

    img = cv2.line(img, tuple(axes[0]), tuple(axes[1]), (0, 0, 255), 3, lineType=cv2.LINE_AA)
    img = cv2.line(img, tuple(axes[0]), tuple(axes[3]), (255, 0, 0), 3, lineType=cv2.LINE_AA)
    img = cv2.line(img, tuple(axes[0]), tuple(axes[2]), (0, 255, 0), 3, lineType=cv2.LINE_AA)  ## y last

    return img
