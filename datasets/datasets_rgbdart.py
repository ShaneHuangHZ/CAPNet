import sys
import os
os.environ['OPENBLAS_NUM_THREADS'] = '64'
import cv2
import random
import torch
import numpy as np
import torch.utils.data as data
import copy
import json
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.data_augmentation import defor_2D
from utils.data_augmentation import data_augment
from utils.datasets_utils import aug_bbox_DZI, get_2d_coord_np, crop_resize_by_warp_affine
from utils.sgpa_utils import get_bbox
from utils.transforms import *
from utils.read_utils import load_anno_dict,load_depth_map_direct
from cutoop.data_loader import Dataset
from cutoop.eval_utils import *
from cutoop.transform import *
from torch_cluster import radius_graph


def erode_point_cloud_gpu(points, radius=0.05, min_neighbors=6):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tensor = torch.tensor(points, device=device)
    
    # 构建邻接关系
    row = radius_graph(tensor, r=radius, loop=False)
    
    # 统计邻域计数
    unique, counts = torch.unique(row[1], return_counts=True)
    counts_all = torch.zeros(len(tensor), device=device)
    counts_all[unique] = counts.float()
    
    # 应用过滤
    mask = counts_all >= min_neighbors
    return tensor[mask].cpu().numpy(),mask.cpu().numpy()

def pixel2xyz_yoeo(h: int, w: int, pixel: ndarray, intrinsics_fx: float, intrinsics_fy: float, intrinsics_cx: float, intrinsics_cy: float):
    """
    Transform `(pixel[0], pixel[1])` to normalized 3D vector under cv space, using camera intrinsics.

    :param h: height of the actual image
    :param w: width of the actual image
    """

    # scale camera parameters
    # scale_x = w / intrinsics.width
    # scale_y = h / intrinsics.height
    scale_x = 1
    scale_y = 1
    fx = intrinsics_fx * scale_x
    fy = intrinsics_fy * scale_y
    x_offset = intrinsics_cx * scale_x
    y_offset = intrinsics_cy * scale_y

    x = (pixel[1] - x_offset) / fx
    y = (pixel[0] - y_offset) / fy
    vec = np.array([x, y, 1])
    return vec / np.linalg.norm(vec)

def FindMaxDis(pointcloud):
    max_xyz = pointcloud.max(0)
    min_xyz = pointcloud.min(0)
    center = (max_xyz + min_xyz) / 2
    max_radius = ((((pointcloud - center)**2).sum(1))**0.5).max()
    return max_radius, center

def WorldSpaceToBallSpace(pointcloud):
    """
    change the raw pointcloud in world space to united vector ball space
    return: max_radius: the max_distance in raw pointcloud to center
            center: [x,y,z] of the raw center
    """
    max_radius, center = FindMaxDis(pointcloud)  
    pointcloud_normalized = (pointcloud - center) / max_radius 
    return pointcloud_normalized, max_radius, center

def getInstanceInfo(xyz, instance_label):
    instance_info = np.ones((xyz.shape[0], 9), dtype=np.float32) * -100.0
    instance_pointnum = []
    instance_num = int(instance_label.max()) + 1
    for i_ in range(instance_num):
        inst_idx_i = np.where(instance_label == i_)[0]
        xyz_i = xyz[inst_idx_i]
        min_xyz_i = xyz_i.min(0)
        max_xyz_i = xyz_i.max(0)
        mean_xyz_i = xyz_i.mean(0)
        instance_info_i = instance_info[inst_idx_i]
        instance_info_i[:, 0:3] = mean_xyz_i
        instance_info_i[:, 3:6] = min_xyz_i
        instance_info_i[:, 6:9] = max_xyz_i
        instance_info[inst_idx_i] = instance_info_i

        instance_pointnum.append(inst_idx_i.size)

    return instance_num, {"instance_info": instance_info, "instance_pointnum": instance_pointnum}



class RGBDArtDataSet(data.Dataset):
    def __init__(self, 
                 cfg,
                 dynamic_zoom_in_params,
                 deform_2d_params,
                 source=None, 
                 mode='train', 
                 data_dir=None,
                 n_pts=1024, 
                 img_size=224, 
                 per_obj='',
                 ):
        
        self.cfg = cfg
        self.source = source
        self.mode = mode
        self.data_dir = data_dir
        self.n_pts = n_pts
        self.img_size = img_size
        self.dynamic_zoom_in_params = dynamic_zoom_in_params
        self.deform_2d_params = deform_2d_params
        directory_yoeo=os.path.join(
            self.data_dir,
            (mode), 
            'rgb'
        )
        img_list = [os.path.splitext(os.path.join(directory_yoeo, file))[0] for file in os.listdir(directory_yoeo) if os.path.isfile(os.path.join(directory_yoeo, file))]
        assert len(img_list)
        self.img_list = img_list
        self.per_obj = per_obj
        self.per_obj_id = None
        self.length = len(self.img_list)
        self.REPCNT = 1 if mode == 'train' and not cfg.load_per_object else 1
        self.length *= self.REPCNT
        assert not self.per_obj or not cfg.load_per_object # for simplicity, not supported together
    
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img_path = self.img_list[index // self.REPCNT]
        filename = os.path.basename(img_path)
        directory = img_path.rsplit('/', 2)[0] + '/' 
        anno=load_anno_dict(directory,filename) 
        img_path=img_path.replace('rgb', 'metafile')
        with open(img_path + ".json", 'r') as f:
            gts  = json.load(f)
        img_path=img_path.replace('metafile', 'rgb')
        rgb = Dataset.load_color(img_path + ".png")
        depth_path=img_path.replace('rgb', 'depth')
        scale_path=img_path.replace('rgb', 'scale')+".txt"
        depth = load_depth_map_direct(depth_path+ ".npz")
        depth[depth > 1e3] = 0
        mask=anno['semantic_segmentation'].copy()
        mask=mask>-2

        if not (mask.shape[:2] == depth.shape[:2] == rgb.shape[:2]):
            assert 0, "invalid data"
        intrinsics = gts['camera_intrinsic']
        original_shape = (3, 3)
        mat_K=np.array(intrinsics).reshape(original_shape)
        intrinsics_fx=mat_K[0,0]
        intrinsics_fy=mat_K[1,1]
        intrinsics_cx=mat_K[0,2]
        intrinsics_cy=mat_K[1,2]
        npcs=anno['npcs_map']
        inst=anno['instance_segmentation']
        sem=anno['semantic_segmentation']
        im_H, im_W = rgb.shape[0], rgb.shape[1]
        coord_2d = get_2d_coord_np(im_W, im_H).transpose(1, 2, 0) # xy map
        object_mask = mask
        ys, xs = np.argwhere(object_mask).transpose(1, 0)

        rmin, rmax, cmin, cmax = np.min(ys), np.max(ys), np.min(xs), np.max(xs)
        rmin, rmax, cmin, cmax = get_bbox([rmin, cmin, rmax, cmax], im_H, im_W)
        bbox_xyxy = np.array([cmin, rmin, cmax, rmax])
        bbox_center, scale = aug_bbox_DZI(self.dynamic_zoom_in_params, bbox_xyxy, im_H, im_W)

        roi_coord_2d = crop_resize_by_warp_affine(
            coord_2d, bbox_center, scale, self.img_size, interpolation=cv2.INTER_NEAREST
        ).transpose(2, 0, 1)

        roi_rgb_ = crop_resize_by_warp_affine(
            rgb, bbox_center, scale, self.img_size, interpolation=cv2.INTER_LINEAR
        )
        roi_npcs_ = crop_resize_by_warp_affine(
            npcs, bbox_center, scale, self.img_size, interpolation=cv2.INTER_NEAREST
        )
        roi_sem_ = crop_resize_by_warp_affine(
            sem, bbox_center, scale, self.img_size, interpolation=cv2.INTER_NEAREST
        )
        roi_inst_ = crop_resize_by_warp_affine(
            inst, bbox_center, scale, self.img_size, interpolation=cv2.INTER_NEAREST
        )
        roi_rgb = RGBDArtDataSet.rgb_transform(roi_rgb_)
        mask_target = mask.copy().astype(np.float32)
        roi_mask = crop_resize_by_warp_affine(
            mask_target, bbox_center, scale, self.img_size, interpolation=cv2.INTER_NEAREST
        )
        roi_mask = np.expand_dims(roi_mask, axis=0)
        roi_depth = crop_resize_by_warp_affine(
            depth, bbox_center, scale, self.img_size, interpolation=cv2.INTER_NEAREST
        )
        roi_depth = np.expand_dims(roi_depth, axis=0)
        depth_valid = roi_depth > 0
        if np.sum(depth_valid) <= 1.0:
            return self.__getitem__((index + 1) % self.__len__())
        roi_m_d_valid = roi_mask.astype(np.bool_) * depth_valid
        if np.sum(roi_m_d_valid) <= 1.0:
            return self.__getitem__((index + 1) % self.__len__())
        roi_mask_def = defor_2D(
            roi_mask, 
            rand_r=self.deform_2d_params['roi_mask_r'], 
            rand_pro=self.deform_2d_params['roi_mask_pro']
        )
        valid = (np.squeeze(roi_depth, axis=0) > 0) * roi_mask_def > 0 
        valid_ins=roi_inst_!=-2
        valid_sem=roi_sem_!=-2
        valid=valid&valid_ins&valid_sem
        xs, ys = np.argwhere(valid).transpose(1, 0)
        valid = valid.reshape(-1)
        pcl_in,pcl_rgb = self._depth_to_pcl(roi_rgb_,roi_depth, mat_K, roi_coord_2d, valid)
        if len(pcl_in) < 50:
            return self.__getitem__((index + 1) % self.__len__())
        
        _, erode_mask = erode_point_cloud_gpu(pcl_in, radius=0.05, min_neighbors=6)
        pcl_rgb=pcl_rgb[erode_mask]
        pcl_in=pcl_in[erode_mask]

        ids, pcl_in = self._sample_points(pcl_in, self.n_pts)


        pcl_in, max_radius, center = WorldSpaceToBallSpace(pcl_in)
        scale_param = np.array([max_radius, center[0], center[1], center[2]])
        os.makedirs(os.path.dirname(scale_path), exist_ok=True)
        np.savetxt(scale_path, scale_param, fmt='%.6f')  # 格式化为 6 位小数
        pcl_rgb=pcl_rgb[ids]
        xs, ys = xs[ids], ys[ids]
        roi_npcs=roi_npcs_.reshape(-1,3)[valid][ids]
        roi_sem=roi_sem_.reshape(-1)[valid][ids]
        roi_sem=roi_sem+1
        roi_inst=roi_inst_.reshape(-1)[valid][ids]
        mask_inst = roi_inst == -1
        roi_inst[mask_inst] = -100
        j = 0
        while (j < roi_inst.max()):
            if (len(np.where(roi_inst == j)[0]) == 0):
                mask = roi_inst == roi_inst.max()
                roi_inst[mask] = j
            j += 1
        _, inst_infos = getInstanceInfo(pcl_in,roi_inst.astype(np.int32))
        inst_info = inst_infos["instance_info"]
        gt_offsets = inst_info[:,:3] - pcl_in
        data_dict = {}
        data_dict['pcl_in'] = torch.as_tensor(pcl_in.astype(np.float32)).contiguous()
        data_dict['pcl_rgb'] = torch.as_tensor(pcl_rgb.astype(np.float32)).contiguous()
        data_dict['file_name'] = filename
        data_dict['roi_npcs'] = torch.as_tensor(np.ascontiguousarray(roi_npcs), dtype=torch.float32).contiguous()
        data_dict['gt_offsets'] = torch.as_tensor(np.ascontiguousarray(gt_offsets), dtype=torch.float32).contiguous()
        data_dict['roi_sem'] = torch.as_tensor(np.ascontiguousarray(roi_sem), dtype=torch.int64).contiguous()
        data_dict['roi_inst'] = torch.as_tensor(np.ascontiguousarray(roi_inst), dtype=torch.int8).contiguous()
        data_dict['roi_rgb_'] = torch.as_tensor(np.ascontiguousarray(roi_rgb_), dtype=torch.uint8).contiguous()
        data_dict['roi_rgb'] = torch.as_tensor(np.ascontiguousarray(roi_rgb), dtype=torch.float32).contiguous()

        data_dict['roi_xs'] = torch.as_tensor(np.ascontiguousarray(xs), dtype=torch.int64).contiguous()
        data_dict['roi_ys'] = torch.as_tensor(np.ascontiguousarray(ys), dtype=torch.int64).contiguous()
        data_dict['roi_center_dir'] = torch.as_tensor(pixel2xyz_yoeo(im_H, im_W, bbox_center, intrinsics_fx,intrinsics_fy,intrinsics_cx,intrinsics_cy), dtype=torch.float32).contiguous()
        intrinsics_list = [intrinsics_fx, intrinsics_fy, intrinsics_cx, intrinsics_cy, im_W, im_H]
        data_dict['intrinsics'] = torch.as_tensor(intrinsics_list, dtype=torch.float32).contiguous()
        return data_dict

    def _sample_points(self, pcl, n_pts):
        """ Down sample the point cloud.
        TODO: use farthest point sampling

        Args:
            pcl (torch tensor or numpy array):  NumPoints x 3
            num (int): target point number
        """
        total_pts_num = pcl.shape[0]
        if total_pts_num < n_pts:
            pcl = np.concatenate([np.tile(pcl, (n_pts // total_pts_num, 1)), pcl[:n_pts % total_pts_num]], axis=0)
            ids = np.concatenate([np.tile(np.arange(total_pts_num), n_pts // total_pts_num), np.arange(n_pts % total_pts_num)], axis=0)
        else:
            ids = np.random.permutation(total_pts_num)[:n_pts]
            pcl = pcl[ids]
        return ids, pcl
    
    def _depth_to_pcl(self, rgb,depth, K, xymap, valid):
        K = K.reshape(-1)
        cx, cy, fx, fy = K[2], K[5], K[0], K[4]
        depth = depth.reshape(-1).astype(np.float32)[valid]

        rgb_map=np.transpose(rgb, (2, 0, 1)) / 255.0
        red=rgb_map[0].reshape(-1)[valid]
        green=rgb_map[1].reshape(-1)[valid]
        blue=rgb_map[2].reshape(-1)[valid]
        pcl_rgb = np.stack((red, green, blue), axis=-1)
        # import pdb
        # pdb.set_trace()
        x_map = xymap[0].reshape(-1)[valid]
        y_map = xymap[1].reshape(-1)[valid]
        real_x = (x_map - cx) * depth / fx
        real_y = (y_map - cy) * depth / fy
        pcl = np.stack((real_x, real_y, depth), axis=-1)
        return pcl.astype(np.float32),pcl_rgb.astype(np.float32)

    def rgb_transform(rgb):
        rgb_ = np.transpose(rgb, (2, 0, 1)) / 255
        _mean = (0.485, 0.456, 0.406)
        _std = (0.229, 0.224, 0.225)
        for i in range(3):
            rgb_[i, :, :] = (rgb_[i, :, :] - _mean[i]) / _std[i]
        return rgb_


def get_data_loaders(
    cfg,
    batch_size,
    seed,
    dynamic_zoom_in_params,
    deform_2d_params,
    percentage_data=1.0,
    data_path=None,
    source='None',
    mode='train',
    n_pts=1024,
    img_size=224,
    per_obj='',
    num_workers=32,
):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    dataset = RGBDArtDataSet(
        cfg=cfg,
        dynamic_zoom_in_params=dynamic_zoom_in_params,
        deform_2d_params=deform_2d_params,
        source=source,
        mode=mode,
        data_dir=data_path,
        n_pts=n_pts,
        img_size=img_size,
        per_obj=per_obj,
    )
    npart_path = os.path.join(cfg.data_path, mode, 'npart.txt')
    filtered_indices = list(range(len(dataset)))
    print(f"npart.txt not found at {npart_path}. Skipping loading.")
    dataset = torch.utils.data.Subset(dataset, filtered_indices)
    size = int(percentage_data * len(dataset))
    dataset, _ = torch.utils.data.random_split(dataset, (size, len(dataset) - size))

    if mode =='train':
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False if mode != 'train' else None,
        sampler=train_sampler if mode == 'train' else None,
        num_workers=num_workers,
        persistent_workers=True,
        drop_last=False,
        pin_memory=True,
    )
    return dataloader



def get_data_loaders_from_cfg(cfg, data_type=['train', 'val', 'test']):
    data_loaders = {}

    if 'test_inter' in data_type:
        test_inter_loader = get_data_loaders(
            cfg=cfg,
            batch_size=cfg.batch_size, 
            seed=cfg.seed,
            dynamic_zoom_in_params=cfg.DYNAMIC_ZOOM_IN_PARAMS,
            deform_2d_params=cfg.DEFORM_2D_PARAMS,
            percentage_data=cfg.percentage_data_for_test,            
            data_path=cfg.data_path,
            source=cfg.test_source,
            mode='test_inter',
            n_pts=cfg.num_points,
            img_size=cfg.img_size,
            per_obj=cfg.per_obj,
            num_workers=cfg.num_workers,
        )
        data_loaders['test_inter_loader'] = test_inter_loader

    if 'test_intra' in data_type:
        test_intra_loader = get_data_loaders(
            cfg=cfg,
            batch_size=cfg.batch_size, 
            seed=cfg.seed,
            dynamic_zoom_in_params=cfg.DYNAMIC_ZOOM_IN_PARAMS,
            deform_2d_params=cfg.DEFORM_2D_PARAMS,
            percentage_data=cfg.percentage_data_for_test,            
            data_path=cfg.data_path,
            source=cfg.test_source,
            mode='test_intra',
            n_pts=cfg.num_points,
            img_size=cfg.img_size,
            per_obj=cfg.per_obj,
            num_workers=cfg.num_workers,
        )
        data_loaders['test_intra_loader'] = test_intra_loader

    return data_loaders


def process_batch(batch_sample,
                  device,
                  pose_mode='quat_wxyz',
                  PTS_AUG_PARAMS=None):
    if PTS_AUG_PARAMS==None:
        PC_da = batch_sample['pcl_in'].to(device)
        PC_da_rgb = batch_sample['pcl_rgb'].to(device)
    elif 'old_sym_info' in batch_sample: # NOCS augmentation
        PC_da, gt_R_da, gt_t_da, gt_s_da = data_augment(
            pts_aug_params=PTS_AUG_PARAMS,
            PC=batch_sample['pcl_in'].to(device), 
            gt_R=batch_sample['rotation'].to(device), 
            gt_t=batch_sample['translation'].to(device),
            gt_s=batch_sample['fsnet_scale'].to(device), 
            mean_shape=batch_sample['mean_shape'].to(device),
            sym=batch_sample['old_sym_info'].to(device),
            aug_bb=batch_sample['aug_bb'].to(device), 
            aug_rt_t=batch_sample['aug_rt_t'].to(device),
            aug_rt_r=batch_sample['aug_rt_R'].to(device),
            model_point=batch_sample['model_point'].to(device), 
            nocs_scale=batch_sample['nocs_scale'].to(device),
            obj_ids=batch_sample['cat_id'].to(device), 
        )
    else:
        PC_da = batch_sample['pcl_in'].to(device)
        PC_da_rgb = batch_sample['pcl_rgb'].to(device)
        
    processed_sample = {}

    processed_sample['roi_npcs'] = batch_sample['roi_npcs'].to(device) 
    processed_sample['roi_sem'] = batch_sample['roi_sem'].to(device) 
    processed_sample['roi_inst'] = batch_sample['roi_inst'].to(device)
    processed_sample['gt_offsets'] = batch_sample['gt_offsets'].to(device)
    processed_sample['file_name'] = batch_sample['file_name']
    

    processed_sample['pts'] = PC_da            
    processed_sample['pts_color'] = PC_da_rgb     
    processed_sample['roi_rgb'] = batch_sample['roi_rgb'].to(device)
    processed_sample['roi_rgb_'] = batch_sample['roi_rgb_'].to(device) 
    assert processed_sample['roi_rgb'].shape[-1] == processed_sample['roi_rgb'].shape[-2]
    assert processed_sample['roi_rgb'].shape[-1] % 14 == 0
    processed_sample['roi_xs'] = batch_sample['roi_xs'].to(device) 
    processed_sample['roi_ys'] = batch_sample['roi_ys'].to(device)
    processed_sample['roi_center_dir'] = batch_sample['roi_center_dir'].to(device) 
    if 'axes_training' in batch_sample:
        processed_sample['axes_training'] = batch_sample['axes_training'].to(device)
        processed_sample['length_training'] = batch_sample['length_training'].to(device)
    """ zero center """
    num_pts = processed_sample['pts'].shape[1]
    zero_mean = torch.mean(processed_sample['pts'][:, :, :3], dim=1)
    processed_sample['zero_mean_pts'] = copy.deepcopy(processed_sample['pts'])
    processed_sample['zero_mean_pts'][:, :, :3] -= zero_mean.unsqueeze(1).repeat(1, num_pts, 1)
    processed_sample['pts_center'] = zero_mean

    return processed_sample 
    