"""
    Evaluation-related codes are modified from
    https://github.com/hughw19/NOCS_CVPR2019
    https://github.com/hetolin/SAR-Net
"""
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from tqdm import tqdm
from utils.symmetry import get_symmetry_tfs


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
    new_coordinates = new_coordinates[:3, :] / new_coordinates[3, :]
    return new_coordinates


def compute_3d_IoU(sRT_1, sRT_2, size_1, size_2, class_name_1, class_name_2, handle_visibility):
    """ Computes IoU overlaps between two 3D bboxes. """
    def asymmetric_3d_iou(sRT_1, sRT_2, size_1, size_2):
        noc_cube_1 = get_3d_bbox(size_1, 0) #(3,N )
        bbox_3d_1 = transform_coordinates_3d(noc_cube_1, sRT_1) #(3,N )
        noc_cube_2 = get_3d_bbox(size_2, 0) #(3,N )
        bbox_3d_2 = transform_coordinates_3d(noc_cube_2, sRT_2) #(3,N )

        # bbox_1_max = np.amax(bbox_3d_1, axis=0) # N
        # bbox_1_min = np.amin(bbox_3d_1, axis=0)
        # bbox_2_max = np.amax(bbox_3d_2, axis=0)
        # bbox_2_min = np.amin(bbox_3d_2, axis=0)
        '''modified'''
        bbox_1_max = np.amax(bbox_3d_1, axis=1) # N
        bbox_1_min = np.amin(bbox_3d_1, axis=1)
        bbox_2_max = np.amax(bbox_3d_2, axis=1)
        bbox_2_min = np.amin(bbox_3d_2, axis=1)

        overlap_min = np.maximum(bbox_1_min, bbox_2_min) # N
        overlap_max = np.minimum(bbox_1_max, bbox_2_max)

        # intersections and union
        if np.amin(overlap_max - overlap_min) < 0:
            intersections = 0
        else:
            intersections = np.prod(overlap_max - overlap_min)
        union = np.prod(bbox_1_max - bbox_1_min) + np.prod(bbox_2_max - bbox_2_min) - intersections
        overlaps = intersections / union
        return overlaps

    if sRT_1 is None or sRT_2 is None:
        return -1
    max_iou = asymmetric_3d_iou(sRT_1, sRT_2, size_1, size_2)

    return max_iou


def compute_IoU_matches(gt_class_ids, gt_sRT, gt_size, gt_handle_visibility,
                        pred_class_ids, pred_sRT, pred_size, pred_scores,
                        synset_names, iou_3d_thresholds, score_threshold=0):

    num_pred = len(pred_class_ids)
    num_gt = len(gt_class_ids) 
    indices = np.zeros(0)

    if num_pred:
        indices = np.argsort(-pred_scores,kind='mergesort') 
        pred_class_ids = pred_class_ids[indices].copy()
        pred_size = pred_size[indices].copy()
        pred_sRT = pred_sRT[indices].copy()
    overlaps = np.zeros((num_pred, num_gt), dtype=np.float32)
    for i in range(num_pred):
        for j in range(num_gt):
            overlaps[i, j] = compute_3d_IoU(pred_sRT[i], gt_sRT[j], pred_size[i, :], gt_size[j],            
                synset_names[pred_class_ids[i]], synset_names[gt_class_ids[j]], gt_handle_visibility[j])
    num_iou_3d_thres = len(iou_3d_thresholds) 
    pred_matches = -1 * np.ones([num_iou_3d_thres, num_pred])
    gt_matches = -1 * np.ones([num_iou_3d_thres, num_gt])
    for s, iou_thres in enumerate(iou_3d_thresholds):
        for i in range(indices.shape[0]):
            sorted_ixs = np.argsort(-overlaps[i]) 
            low_score_idx = np.where(overlaps[i, sorted_ixs] < score_threshold)[0]
            if low_score_idx.size > 0:
                sorted_ixs = sorted_ixs[:low_score_idx[0]]  
            for j in sorted_ixs:  
                if gt_matches[s, j] > -1:
                    continue
                iou = overlaps[i, j]
                if iou < iou_thres: 
                    break
                if not pred_class_ids[i] == gt_class_ids[j]: 
                    continue
                if iou > iou_thres:       
                    gt_matches[s, j] = i  
                    pred_matches[s, i] = j
                    break
    return gt_matches, pred_matches, overlaps, indices   

def compute_RT_errors(sRT_1, sRT_2, class_id, handle_visibility, synset_names):
    """
    Args:
        sRT_1: [4, 4]. homogeneous affine transformation
        sRT_2: [4, 4]. homogeneous affine transformation

    Returns:
        theta: angle difference of R in degree
        shift: l2 difference of T in centimeter
    """
    # make sure the last row is [0, 0, 0, 1]
    if sRT_1 is None or sRT_2 is None:
        return -1
    try:
        assert np.array_equal(sRT_1[3, :], sRT_2[3, :])
        assert np.array_equal(sRT_1[3, :], np.array([0, 0, 0, 1]))
    except AssertionError:
        print(sRT_1[3, :], sRT_2[3, :])
        exit()
    R_z_90 = np.array([[0, -1, 0],
                   [1,  0, 0],
                   [0,  0, 1]])
    R_z_180 = np.array([[-1, 0, 0],
                    [0, -1, 0],
                    [0, 0, 1]])
    R_x_90 = np.array([[1, 0, 0],
                [0,  0, -1],
                [0,  1, 0]])
    R_x_180=np.array([[1, 0, 0],
                [0,  -1, 0],
                [0,  1, 0]])
    R_y_90 = np.array([[0, 0, 1],
                [0,  1, 0],
                [-1,  0, 0]])
    R1 = sRT_1[:3, :3] / np.cbrt(np.linalg.det(sRT_1[:3, :3]))
    T1 = sRT_1[:3, 3]
    R2 = sRT_2[:3, :3] / np.cbrt(np.linalg.det(sRT_2[:3, :3]))
    T2 = sRT_2[:3, 3]
    tfs=get_symmetry_tfs(synset_names[class_id])
    min_theta = float('inf')
    if synset_names[class_id] in ["mug"
    ]:
        z = np.array([0, 0, 1])
        y1 = R1 @ z
        y2 = R2 @ z
        cos_theta = y1.dot(y2) / (np.linalg.norm(y1) * np.linalg.norm(y2))
        min_theta = np.arccos(np.clip(cos_theta, -1.0, 1.0)) * 180 / np.pi

    else:
        for tf in tfs:
            tf_rotation = tf[:3, :3]  
            R1_transformed =  R1@tf_rotation  
            R = R1_transformed @ R2.transpose()
            cos_theta = (np.trace(R) - 1) / 2
            theta = np.arccos(np.clip(cos_theta, -1.0, 1.0)) * 180 / np.pi
            if theta < min_theta:
                min_theta = theta

    theta=min_theta
    shift = np.linalg.norm(T1 - T2)
    result = np.array([theta, shift])

    return result

def compute_size_errors(size1, size2):
    diff = size1 - size2 
    abs_diff = np.abs(diff) 
    xyz_loss = np.mean(abs_diff) 
    return xyz_loss


def compute_RT_overlaps(gt_class_ids, gt_sRT, gt_handle_visibility, pred_class_ids, pred_sRT, synset_names):
    """ Finds overlaps between prediction and ground truth instances.

    Returns:
        overlaps:

    """
    num_pred = len(pred_class_ids)
    num_gt = len(gt_class_ids)
    overlaps = np.zeros((num_pred, num_gt, 2))
    for i in range(num_pred):
        for j in range(num_gt):
            try:
                overlaps[i, j, :] = compute_RT_errors(np.array(pred_sRT[i], dtype=np.float64), np.array(gt_sRT[j], dtype=np.float64), gt_class_ids[j], gt_handle_visibility[j], synset_names)
            except Exception as e:
                print(f"Error occurred at i={i}, j={j}: {e}")
                import pdb
                pdb.set_trace()  # 进入 pdb 调试
    return overlaps

def compute_size_overlaps(gt_class_ids, gt_size, gt_handle_visibility, pred_class_ids, pred_size, synset_names):
    """ Finds overlaps between prediction and ground truth instances.

    Returns:
        overlaps:

    """
    num_pred = len(pred_class_ids)
    num_gt = len(gt_class_ids)
    overlaps = np.zeros((num_pred, num_gt, 1))
    for i in range(num_pred):
        for j in range(num_gt):
            try:
                overlaps[i, j, :] = compute_size_errors(pred_size[i], gt_size[j])
            except Exception as e:
                print(f"Error occurred at i={i}, j={j}: {e}")
                import pdb
                pdb.set_trace()
    return overlaps

def compute_RT_matches(overlaps, pred_class_ids, gt_class_ids, degree_thres_list, shift_thres_list):
    num_degree_thres = len(degree_thres_list)
    num_shift_thres = len(shift_thres_list)
    num_pred = len(pred_class_ids)
    num_gt = len(gt_class_ids)

    pred_matches = -1 * np.ones((num_degree_thres, num_shift_thres, num_pred))
    gt_matches = -1 * np.ones((num_degree_thres, num_shift_thres, num_gt))

    if num_pred == 0 or num_gt == 0:
        return gt_matches, pred_matches

    assert num_pred == overlaps.shape[0]
    assert num_gt == overlaps.shape[1]
    assert overlaps.shape[2] == 2

    for d, degree_thres in enumerate(degree_thres_list):
        for s, shift_thres in enumerate(shift_thres_list):
            for i in range(num_pred): 
                sum_degree_shift = overlaps[i, :, :][:,1]
                sorted_ixs = np.argsort(sum_degree_shift)  
                for j in sorted_ixs:     
                    if gt_matches[d, s, j] > -1 or pred_class_ids[i] != gt_class_ids[j]:
                        continue
                    if overlaps[i, j, 0] > degree_thres or overlaps[i, j, 1] > shift_thres:
                        continue
                    gt_matches[d, s, j] = i   
                    pred_matches[d, s, i] = j 
                    break

    return gt_matches, pred_matches


def compute_ap_and_acc(pred_matches, pred_scores, gt_matches):
    assert pred_matches.shape[0] == pred_scores.shape[0]
    score_indices = np.argsort(-pred_scores)
    pred_matches = pred_matches[score_indices]
    precisions = np.cumsum(pred_matches > -1) / (np.arange(len(pred_matches)) + 1)
    recalls = np.cumsum(pred_matches > -1).astype(np.float32) / len(gt_matches)
    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])
    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
    ap = np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])
    if(len(pred_matches) <= 0):
        import pdb
        pdb.set_trace()
    assert len(pred_matches) > 0
    acc = np.sum(pred_matches > -1) / len(pred_matches)

    return ap, acc


def compute_mAP(pred_results, out_dir, degree_thresholds=[180], shift_thresholds=[100],
                iou_3d_thresholds=[0.1], iou_pose_thres=0.1, use_matches_for_pose=False):
    """ Compute mean Average Precision.

    Returns:
        iou_aps:
        pose_aps:
        iou_acc:
        pose_acc:

    """
    synset_names = ["BG","line_fixed_handle", "round_fixed_handle", "slider_button", "hinge_door", "slider_drawer", "slider_lid",
    "hinge_lid", "hinge_knob", "hinge_handle"]  
    num_classes=len(synset_names) 
    degree_thres_list = list(degree_thresholds)
    num_degree_thres = len(degree_thres_list)  

    shift_thres_list = list(shift_thresholds)
    num_shift_thres = len(shift_thres_list)
    iou_thres_list = list(iou_3d_thresholds)
    num_iou_thres = len(iou_thres_list)

    if use_matches_for_pose:
        assert iou_pose_thres in iou_thres_list
    iou_thres_index=iou_thres_list.index(iou_pose_thres)

    iou_aps = np.zeros((num_classes + 1, num_iou_thres))
    iou_acc = np.zeros((num_classes + 1, num_iou_thres))
    iou_pred_matches_all = [np.zeros((num_iou_thres, 30000)) for _ in range(num_classes)]
    iou_pred_scores_all = [np.zeros((num_iou_thres, 30000)) for _ in range(num_classes)]
    iou_gt_matches_all = [np.zeros((num_iou_thres, 30000)) for _ in range(num_classes)]
    iou_pred_count = [0 for _ in range(num_classes)]
    iou_gt_count = [0 for _ in range(num_classes)]

    # pose
    pose_aps = np.zeros((num_classes + 1, num_degree_thres, num_shift_thres))
    pose_acc = np.zeros((num_classes + 1, num_degree_thres, num_shift_thres))
    #
    pose_pred_matches_all = [np.zeros((num_degree_thres, num_shift_thres, 30000)) for _ in range(num_classes)]
    pose_pred_scores_all = [np.zeros((num_degree_thres, num_shift_thres, 30000)) for _ in range(num_classes)]
    pose_gt_matches_all = [np.zeros((num_degree_thres, num_shift_thres, 30000)) for _ in range(num_classes)]
    pose_pred_count = [0 for _ in range(num_classes)]
    pose_gt_count = [0 for _ in range(num_classes)]
    RT_overlaps_sum=np.zeros((num_classes,1,2))
    size_error_sum=np.zeros((num_classes,1,1))

    RT_count=np.zeros((num_classes,1))

    miou_count=np.zeros((num_classes,1))
    iou_sum=np.zeros((num_classes,1))

    for _, result in enumerate(tqdm(pred_results)):
        gt_class_ids = result['gt_class_ids'].flatten().astype(np.int32)
        gt_sRT = np.array(result['gt_RTs'])
        gt_size = np.array(result['gt_scales'])
        gt_handle_visibility = result['gt_handle_visibility']


        pred_class_ids = result['pred_class_ids'].flatten().astype(np.int32)
        pred_sRT = np.array(result['pred_RTs'])
        pred_size = result['pred_scales']

        pred_scores=  result['pred_scores'].flatten().astype(np.int32)
        if len(gt_class_ids) == 0 and len(pred_class_ids) == 0:
            continue

        for cls_id in range(1, num_classes):
            cls_gt_class_ids = gt_class_ids[gt_class_ids==cls_id] if len(gt_class_ids) else np.zeros(0)
            cls_gt_sRT = gt_sRT[gt_class_ids==cls_id] if len(gt_class_ids) else np.zeros((0, 4, 4))
            cls_gt_size = gt_size[gt_class_ids==cls_id] if len(gt_class_ids) else np.zeros((0, 3))

            if synset_names[cls_id] != 'mug':
                cls_gt_handle_visibility = np.ones_like(cls_gt_class_ids)
            else:
                cls_gt_handle_visibility = gt_handle_visibility[gt_class_ids==cls_id] if len(gt_class_ids) else np.ones(0)

            cls_pred_class_ids = pred_class_ids[pred_class_ids==cls_id] if len(pred_class_ids) else np.zeros(0)
            cls_pred_sRT = pred_sRT[pred_class_ids==cls_id] if len(pred_class_ids) else np.zeros((0, 4, 4))
            cls_pred_size = pred_size[pred_class_ids==cls_id] if len(pred_class_ids) else np.zeros((0, 3))
            cls_pred_scores = pred_scores[pred_class_ids==cls_id] if len(pred_class_ids) else np.zeros(0)

            iou_cls_gt_match, iou_cls_pred_match, vol_overlap, iou_pred_indices = \
                compute_IoU_matches(cls_gt_class_ids, cls_gt_sRT, cls_gt_size, cls_gt_handle_visibility,
                                    cls_pred_class_ids, cls_pred_sRT, cls_pred_size, cls_pred_scores,
                                    synset_names, iou_thres_list)  

            if len(iou_pred_indices): 
                cls_pred_class_ids = cls_pred_class_ids[iou_pred_indices]  
                cls_pred_sRT = cls_pred_sRT[iou_pred_indices]
                cls_pred_scores = cls_pred_scores[iou_pred_indices]
            if use_matches_for_pose:  
                thres_ind = list(iou_thres_list).index(iou_pose_thres)  
                iou_thres_pred_match = iou_cls_pred_match[thres_ind, :] 
                cls_pred_class_ids = cls_pred_class_ids[iou_thres_pred_match > -1] if len(iou_thres_pred_match) > 0 else np.zeros(0)
                cls_pred_size=cls_pred_size[iou_thres_pred_match > -1]
                cls_pred_sRT = cls_pred_sRT[iou_thres_pred_match > -1] if len(iou_thres_pred_match) > 0 else np.zeros((0, 4, 4))
                cls_pred_scores = cls_pred_scores[iou_thres_pred_match > -1] if len(iou_thres_pred_match) > 0 else np.zeros(0)
                iou_thres_gt_match = iou_cls_gt_match[thres_ind, :]  
                cls_gt_size=cls_gt_size[iou_thres_gt_match > -1]
                cls_gt_class_ids = cls_gt_class_ids[iou_thres_gt_match > -1] if len(iou_thres_gt_match) > 0 else np.zeros(0)
                cls_gt_sRT = cls_gt_sRT[iou_thres_gt_match > -1] if len(iou_thres_gt_match) > 0 else np.zeros((0, 4, 4))
                cls_gt_handle_visibility = cls_gt_handle_visibility[iou_thres_gt_match > -1] if len(iou_thres_gt_match) > 0 else np.zeros(0)

            iou_cls_gt_match, iou_cls_pred_match, vol_overlap, iou_pred_indices = \
                compute_IoU_matches(cls_gt_class_ids, cls_gt_sRT, cls_gt_size, cls_gt_handle_visibility,
                                    cls_pred_class_ids, cls_pred_sRT, cls_pred_size, cls_pred_scores,
                                    synset_names, iou_thres_list)  


            num_pred = iou_cls_pred_match.shape[1] 
            pred_start = iou_pred_count[cls_id]  
            pred_end = pred_start + num_pred
            iou_pred_count[cls_id] = pred_end 
            iou_pred_matches_all[cls_id][:, pred_start:pred_end] = iou_cls_pred_match  
            cls_pred_scores_tile = np.tile(cls_pred_scores, (num_iou_thres, 1))
            assert cls_pred_scores_tile.shape[1] == num_pred
            iou_pred_scores_all[cls_id][:, pred_start:pred_end] = cls_pred_scores_tile 


            num_gt = iou_cls_gt_match.shape[1]
            gt_start = iou_gt_count[cls_id]
            gt_end = gt_start + num_gt  
            iou_gt_count[cls_id] = gt_end
            iou_gt_matches_all[cls_id][:, gt_start:gt_end] = iou_cls_gt_match

            if len(vol_overlap)>0 and use_matches_for_pose:
                selected_range = [i for i in range(vol_overlap.shape[0])]
                for idx in selected_range:
                    iou_sum[cls_gt_class_ids[idx]]+=vol_overlap[int(idx),int(iou_cls_pred_match[iou_thres_index][idx])]
                    iou_sum[0]+=vol_overlap[int(idx),int(iou_cls_pred_match[iou_thres_index][idx])]
                    miou_count[cls_gt_class_ids[idx]]+=1
                    miou_count[0]+=1
            RT_count[cls_id]+=len(cls_pred_class_ids)
            RT_count[0]+=len(cls_pred_class_ids)
            RT_overlaps = compute_RT_overlaps(cls_gt_class_ids, cls_gt_sRT, cls_gt_handle_visibility,
                                              cls_pred_class_ids, cls_pred_sRT, synset_names)
            RT_size_overlaps=compute_size_overlaps(cls_gt_class_ids, cls_gt_size, cls_gt_handle_visibility,
                                              cls_pred_class_ids, cls_pred_size, synset_names)
            if len(RT_overlaps)!=0 and use_matches_for_pose:
                for l in range(len(RT_overlaps)):
                    try:
                        if int(iou_thres_pred_match[l]) < 0:
                            continue
                        size_error_sum[cls_id]+=RT_size_overlaps[l,int(iou_cls_pred_match[0][l]), :]
                        RT_overlaps_sum[cls_id]+=RT_overlaps[l,int(iou_cls_pred_match[0][l]), :]
                        size_error_sum[0]+=RT_size_overlaps[l,int(iou_cls_pred_match[0][l]), :]
                        RT_overlaps_sum[0]+=RT_overlaps[l,int(iou_cls_pred_match[0][l]), :]
                    except IndexError as e:
                        import pdb
                        pdb.set_trace()

            pose_cls_gt_match, pose_cls_pred_match = compute_RT_matches(RT_overlaps, cls_pred_class_ids, cls_gt_class_ids,
                                                                        degree_thres_list, shift_thres_list)
            num_pred = pose_cls_pred_match.shape[2]
            pred_start = pose_pred_count[cls_id]
            pred_end = pred_start + num_pred
            pose_pred_count[cls_id] = pred_end
            pose_pred_matches_all[cls_id][:, :, pred_start:pred_end] = pose_cls_pred_match
            cls_pred_scores_tile = np.tile(cls_pred_scores, (num_degree_thres, num_shift_thres, 1))
            assert cls_pred_scores_tile.shape[2] == num_pred
            pose_pred_scores_all[cls_id][:, :, pred_start:pred_end] = cls_pred_scores_tile
            num_gt = pose_cls_gt_match.shape[2]
            gt_start = pose_gt_count[cls_id]
            gt_end = gt_start + num_gt
            pose_gt_count[cls_id] = gt_end
            pose_gt_matches_all[cls_id][:, :, gt_start:gt_end] = pose_cls_gt_match


    # trim zeros
    for cls_id in range(num_classes):
        iou_pred_matches_all[cls_id] = iou_pred_matches_all[cls_id][:, :iou_pred_count[cls_id]]   
        iou_pred_scores_all[cls_id] = iou_pred_scores_all[cls_id][:, :iou_pred_count[cls_id]]
        iou_gt_matches_all[cls_id] = iou_gt_matches_all[cls_id][:, :iou_gt_count[cls_id]]
        # pose
        pose_pred_matches_all[cls_id] = pose_pred_matches_all[cls_id][:, :, :pose_pred_count[cls_id]]
        pose_pred_scores_all[cls_id] = pose_pred_scores_all[cls_id][:, :, :pose_pred_count[cls_id]]
        pose_gt_matches_all[cls_id] = pose_gt_matches_all[cls_id][:, :, :pose_gt_count[cls_id]]

    for cls_id in range(1, num_classes):
        for s, iou_thres in enumerate(iou_thres_list):
            if len(iou_pred_matches_all[cls_id][s, :]) <= 0 or len(iou_gt_matches_all[cls_id][s, :])<=0:
                break

            iou_aps[cls_id, s], iou_acc[cls_id, s] = compute_ap_and_acc(iou_pred_matches_all[cls_id][s, :],
                                                                        iou_pred_scores_all[cls_id][s, :],
                                                                        iou_gt_matches_all[cls_id][s, :])  
    iou_aps[-1, :] = np.mean(iou_aps[1:-1, :], axis=0)
    iou_acc[-1, :] = np.mean(iou_acc[1:-1, :], axis=0)

    # compute pose mAP
    for i, degree_thres in enumerate(degree_thres_list):
        for j, shift_thres in enumerate(shift_thres_list):
            for cls_id in range(1, num_classes):
                cls_pose_pred_matches_all = pose_pred_matches_all[cls_id][i, j, :]
                cls_pose_gt_matches_all = pose_gt_matches_all[cls_id][i, j, :]
                cls_pose_pred_scores_all = pose_pred_scores_all[cls_id][i, j, :]
                if len(cls_pose_pred_matches_all) <= 0 or len(cls_pose_gt_matches_all)<=0:
                    continue
                pose_aps[cls_id, i, j], pose_acc[cls_id, i, j] = compute_ap_and_acc(cls_pose_pred_matches_all,
                                                                                    cls_pose_pred_scores_all,
                                                                                    cls_pose_gt_matches_all)
            pose_aps[-1, i, j] = np.mean(pose_aps[1:-1, i, j])
            pose_acc[-1, i, j] = np.mean(pose_acc[1:-1, i, j])


    RT_count=np.repeat(np.expand_dims(RT_count, axis=2),2, axis=2)
    RT_overlaps_error=RT_overlaps_sum/RT_count
    size_error=size_error_sum/RT_count
    miou=iou_sum/miou_count

    return iou_aps, pose_aps, iou_acc, pose_acc,RT_overlaps_error[0][0],size_error[0][0][0],miou[0][0]
