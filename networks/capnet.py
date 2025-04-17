import sys
import os
import torch
import torch.nn as nn
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from networks.pts_encoder.pointnet2 import Pointnet2MSG
from networks.gf_algorithms.scorenet import PoseNet
from networks.gf_algorithms.sde import init_sde
from configs.config import get_config
from utils.capnet_utils import encode_axes
import torch.nn.functional as F



import torch
import torch.nn.functional as F
sys.path.append('../../sam2')
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

class MainNet(nn.Module):
    dino_name = 'dinov2_vits14'
    dino_dim = 96+384
    embedding_dim = 60
    def __init__(self, cfg, prior_fn, marginal_prob_fn, sde_fn, sampling_eps, T):
        super(MainNet, self).__init__()

        self.cfg = cfg
        self.device = cfg.device
        self.is_testing = False
        
        ''' Load model, define SDE '''
        # init SDE config
        self.prior_fn = prior_fn
        self.marginal_prob_fn = marginal_prob_fn
        self.sde_fn = sde_fn
        self.sampling_eps = sampling_eps
        self.T = T
        
        ''' dino v2 '''
        self.featup = torch.hub.load("mhamilton723/FeatUp", 'dinov2', use_norm=True,trust_repo=True)
        self.featup.requires_grad_(False)

        sam2_checkpoint = '../../sam2/checkpoints/sam2.1_hiera_large.pt'
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=cfg.device)
        self.dino = SAM2ImagePredictor(sam2_model)
        self.dino_dim = MainNet.dino_dim
        self.embedding_dim = MainNet.embedding_dim
        
        ''' encode pts '''
        self.pts_encoder = Pointnet2MSG(self.dino_dim+3)

        per_point_feat = False
        self.pose_net = PoseNet(
            self.marginal_prob_fn, 
            (0 if self.cfg.dino != 'global' else self.dino_dim + self.embedding_dim),
            self.cfg.pose_mode, 
            self.cfg.regression_head, 
            per_point_feat
        )

    def tensor_to_numpy_list(self, tensor_batch: torch.Tensor) -> list:
        if tensor_batch.is_cuda:
            tensor_batch = tensor_batch.cpu()

        np_array_batch = tensor_batch.numpy()

        image_list = [np_array_batch[i] for i in range(np_array_batch.shape[0])]

        return image_list
    
    def extract_pts_feature(self, data):
        """extract the input pointcloud feature

        Args:
            data (dict): batch example without pointcloud feature. {'pts': [bs, num_pts, 3], 'sampled_pose': [bs, pose_dim], 't': [bs, 1]}
        Returns:
            data (dict): batch example with pointcloud feature. {'pts': [bs, num_pts, 3], 'pts_feat': [bs, c], 'sampled_pose': [bs, pose_dim], 't': [bs, 1]}
        """
        pts = data['pts']
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            hr_feats2 = self.featup(data['roi_rgb'])
            feat_fp = F.interpolate(hr_feats2, size=(self.cfg.img_size, self.cfg.img_size), mode='bilinear', align_corners=False)

            self.dino.set_image_batch(self.tensor_to_numpy_list(data['roi_rgb_']))
            lr_feats=self.dino._features["high_res_feats"][0]
            hr_feats=self.dino._features["high_res_feats"][1]
            hr_feat = F.interpolate(hr_feats, size=(self.cfg.img_size, self.cfg.img_size), mode='bilinear', align_corners=False)
            lr_feat = F.interpolate(lr_feats, size=(self.cfg.img_size, self.cfg.img_size), mode='bilinear', align_corners=False)
            combined_feat = torch.cat((lr_feat, hr_feat,feat_fp), dim=1)
        combined_feat=combined_feat.reshape(-1, 384+96, self.cfg.img_size * self.cfg.img_size)
        feat=combined_feat.permute(0, 2, 1)

        xs = data['roi_xs'] // 1
        ys = data['roi_ys'] // 1
        pos = xs * self.cfg.img_size + ys
        pos = torch.unsqueeze(pos, -1).expand(-1, -1, self.dino_dim)
        rgb_feat = torch.gather(feat, 1, pos)
        rgb_feat.requires_grad_(False)
        pts_feat = self.pts_encoder(torch.cat([pts,data['pts_color'],rgb_feat], dim=-1))

        return pts_feat
    
    def forward(self, data, mode='main', init_x=None, T0=None):
        '''
        Args:
            data, dict {
                'pts': [bs, num_pts, 3]
                'pts_feat': [bs, c]
                'sampled_pose': [bs, pose_dim]
                't': [bs, 1]
            }
        '''
        if mode == 'main':
            out_main = self.pose_net(data) # normalisation
            return out_main
        elif mode == 'pts_feature':
            pts_feature = self.extract_pts_feature(data)
            return pts_feature ##torch.Size([16, 1024])
        elif mode == 'rgb_feature':
            if self.cfg.dino != 'global':
                return None
            rgb: torch.Tensor = data['roi_rgb'] # torch.Size([16, 3, 224, 224])
            assert rgb.shape[-1] % 14 == 0 and rgb.shape[-2] % 14 == 0
            assert self.embedding_dim % 6 == 0
            return torch.concat([self.dino(rgb), encode_axes(data['roi_center_dir'], dim=self.embedding_dim // 6)], dim=-1) 
        else:
            raise NotImplementedError


