import os
import glob
import numpy as np
import _pickle as cPickle
import torch
from utils.utils_metric import compute_mAP


def convert_to_numpy(obj):
    if obj is None:
        return None 
    elif isinstance(obj, np.ndarray):
        return obj
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().numpy()  
    elif isinstance(obj, (int, float, bool)):
        return np.array([obj])
    elif isinstance(obj, list):
        return np.array([convert_to_numpy(o) for o in obj], dtype=object)
    else:
        raise TypeError(f"Cannot convert object of type {type(obj)} to numpy array.")

def evaluate():
    degree_thres_list = list(range(0, 185, 5))
    shift_thres_list = [i / 100 for i in range(101)]
    iou_thres_list = [i / 100 for i in range(101)]

    result_folder='path/to/your/result_folder' 
    result_pkl_list = glob.glob(os.path.join(result_folder, 'results_*.pkl')) 
    result_pkl_list = sorted(result_pkl_list) 
    assert len(result_pkl_list)  
    pred_results = []
    for pkl_path in result_pkl_list:
        with open(pkl_path, 'rb') as f:
            result = cPickle.load(f)
            if 'gt_handle_visibility' not in result:
                result['gt_handle_visibility'] = np.ones_like(result['gt_class_ids'])
            else:
                assert len(result['gt_handle_visibility']) == len(result['gt_class_ids']), "{} {}".format(
                    result['gt_handle_visibility'], result['gt_class_ids'])
        if any(value is None or (isinstance(value, list) and any(v is None for v in value)) for value in result.values()):
            continue
        result = {key: value if isinstance(value, str) else convert_to_numpy(value) for key, value in result.items()}
        pred_results.append(result)

    # To be consistent with NOCS, set use_matches_for_pose=True for mAP evaluation
    iou_aps, pose_aps, iou_acc, pose_acc,RT_overlaps_error,size_error,miou = compute_mAP(pred_results, result_folder, degree_thres_list, shift_thres_list,
                                                       iou_thres_list, iou_pose_thres=0.1, use_matches_for_pose=True)

    fw = open('{}/eval_logs.txt'.format(result_folder), 'a')
    iou_25_idx = iou_thres_list.index(0.25)
    iou_50_idx = iou_thres_list.index(0.5)
    iou_75_idx = iou_thres_list.index(0.75)
    degree_05_idx = degree_thres_list.index(5)
    degree_10_idx = degree_thres_list.index(10)
    shift_02_idx = shift_thres_list.index(0.02)
    shift_05_idx = shift_thres_list.index(0.05)
    shift_10_idx = shift_thres_list.index(0.1)
    messages = []
    messages.append('mAP:')
    messages.append('3D IoU at 25: {:.2f}'.format(iou_aps[-1, iou_25_idx] * 100))
    messages.append('3D IoU at 50: {:.2f}'.format(iou_aps[-1, iou_50_idx] * 100))
    messages.append('3D IoU at 75: {:.2f}'.format(iou_aps[-1, iou_75_idx] * 100))
    messages.append('5 degree, 2cm: {:.2f}'.format(pose_aps[-1, degree_05_idx, shift_02_idx] * 100))
    messages.append('5 degree, 5cm: {:.2f}'.format(pose_aps[-1, degree_05_idx, shift_05_idx] * 100))
    messages.append('10 degree, 2cm: {:.2f}'.format(pose_aps[-1, degree_10_idx, shift_02_idx] * 100))
    messages.append('10 degree, 5cm: {:.2f}'.format(pose_aps[-1, degree_10_idx, shift_05_idx] * 100))
    messages.append('10 degree, 10cm: {:.2f}'.format(pose_aps[-1, degree_10_idx, shift_10_idx] * 100))
    messages.append('Acc:')
    messages.append('3D IoU at 25: {:.2f}'.format(iou_acc[-1, iou_25_idx] * 100))
    messages.append('3D IoU at 50: {:.2f}'.format(iou_acc[-1, iou_50_idx] * 100))
    messages.append('3D IoU at 75: {:.2f}'.format(iou_acc[-1, iou_75_idx] * 100))
    messages.append('5 degree, 2cm: {:.2f}'.format(pose_acc[-1, degree_05_idx, shift_02_idx] * 100))
    messages.append('5 degree, 5cm: {:.2f}'.format(pose_acc[-1, degree_05_idx, shift_05_idx] * 100))
    messages.append('10 degree, 2cm: {:.2f}'.format(pose_acc[-1, degree_10_idx, shift_02_idx] * 100))
    messages.append('10 degree, 5cm: {:.2f}'.format(pose_acc[-1, degree_10_idx, shift_05_idx] * 100))
    messages.append('10 degree, 10cm: {:.2f}'.format(pose_acc[-1, degree_10_idx, shift_10_idx] * 100))
    messages.append('Rotation_error: {:.3f}'.format(RT_overlaps_error[0]))
    messages.append('Translation_error: {:.3f}'.format(RT_overlaps_error[1]))
    messages.append('Size_error: {:.3f}'.format(size_error))
    messages.append('mIoU: {:.3f}'.format(miou*100))
    for msg in messages:
        print(msg)
        fw.write(msg + '\n')
    fw.close()

if __name__ == "__main__":
    print('Evaluating ...')
    evaluate()

