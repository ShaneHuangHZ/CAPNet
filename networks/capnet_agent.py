import sys
import os
import torch
import numpy as np
import torch.nn as nn
import _pickle as cPickle
from tensorboardX import SummaryWriter
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from networks.gf_algorithms.sde import init_sde
from networks.capnet import MainNet
import pickle
from utils.read_utils import load_rgb_image,load_meta
from utils.visu_utils import draw_detections_pcs_pred
from utils.capnet_utils import TrainClock
from utils.misc import exists_or_mkdir
from utils.transforms import *
import utils.umeyama_pytorch as umeyama
import utils.meanshift_pytorch as ms
num_classes=10
import yaml
with open('./networks/gf_algorithms/config_capnet.yml', 'r') as ff:
    cfg_bins = yaml.safe_load(ff)
 

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



def get_ckpt_and_writer_path(cfg):
        
        ''' init exp folder and writer '''
        ckpt_path = f'./results/ckpts/{cfg.log_dir}'
        writer_path = f'./results/logs/{cfg.log_dir}' if cfg.use_pretrain == False else f'./results/logs/{cfg.log_dir}_continue'
        
        if cfg.is_train:
            exists_or_mkdir('./results')
            exists_or_mkdir(ckpt_path)
            exists_or_mkdir(writer_path)    
        return ckpt_path, writer_path    


class CapNet(nn.Module):
    def __init__(self, cfg):
        super(CapNet, self).__init__()
        
        self.cfg = cfg
        self.is_testing = False
        self.clock = TrainClock()
        self.pts_feature = False
        if self.cfg.local_rank==0:
            self.model_dir, writer_path = get_ckpt_and_writer_path(self.cfg)
        self.pred_path=(self.cfg.pred_path)
        if self.cfg.is_train and self.cfg.local_rank==0:
            self.writer = SummaryWriter(writer_path)
        
        self.prior_fn, self.marginal_prob_fn, self.sde_fn, self.sampling_eps, self.T = init_sde(self.cfg.sde_mode)
        self.net = self.build_net()

    def get_network(self, name):
        if name == 'CapNet':
            return MainNet(self.cfg, self.prior_fn, self.marginal_prob_fn, self.sde_fn, self.sampling_eps, self.T)

    
    
    def build_net(self):
        net = self.get_network('CapNet')
        net = net.to(self.cfg.device)

        if self.cfg.parallel:
            current_device = torch.cuda.current_device()
            print(f"current_device: {current_device}")
            print("init,",self.cfg.local_rank)
            print("self.cfg.device",self.cfg.device)
            net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net).to(self.cfg.device)
            net = nn.parallel.DistributedDataParallel(net.cuda(), device_ids=[self.cfg.local_rank], output_device=self.cfg.local_rank, find_unused_parameters=True,broadcast_buffers=False)
        return net

            

    def load_ckpt(self, name=None, model_dir=None, model_path=False, load_model_only=False):
        """load checkpoint from saved checkpoint"""
        if not model_path:
            if name == 'latest':
                pass
            elif name == 'best':
                pass
            else:
                name = "ckpt_epoch{}".format(name)

            if model_dir is None:
                load_path = os.path.join(self.model_dir, "{}.pth".format(name))
            else:
                load_path = os.path.join(model_dir, "{}.pth".format(name))
        else:
            load_path = model_dir
        if not os.path.exists(load_path):
            raise ValueError("Checkpoint {} not exists.".format(load_path))
        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.cfg.local_rank}
        checkpoint = torch.load(load_path,map_location=map_location)
        print("Loading checkpoint from {} ...".format(load_path))

        new_state_dict = {}
        for key, value in checkpoint['model_state_dict'].items():
            new_key = key.replace("pose_score_net", "pose_net")
            new_state_dict[new_key] = value

        self.net.load_state_dict(new_state_dict)
        if not load_model_only:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.clock.restore_checkpoint(checkpoint['clock'])




    def evaluate(self, data,load_from_pkl="",visual=False):
        '''
        Args:
            data, dict {
                'pts_feat': [bs, c]
                'rgb_feat': [bs, dino_dim] (optional)
                'gt_pose': [bs, pose_dim]
            }
        '''
        with torch.no_grad():
            load_path = os.path.join(self.cfg.data_path, self.cfg.infer_split).rstrip(os.sep) + os.sep
            rgb_image = load_rgb_image(load_path, data['file_name'][0])
            pred_offset,pred_nocs,pred_sem = self.net(data)
            pred_offset.permute(0, 2, 1)[0,:,:]
            _, classes_rgbd = torch.max(pred_sem.permute(0, 2, 1)[0,:,:], 1)
            all_obsv_pcd = data['pts'][0]
            instance_labels=data['roi_inst'][0]
            sem_labels=data['roi_sem'][0]
            gt_npcs=data['roi_npcs'][0]
            num_insts,cout=torch.unique(instance_labels, return_counts=True)
            mask2 = cout > 5
            filtered_elements = num_insts[mask2]
            num_insts=len(filtered_elements)-1
            gt_class_ids=[]
            gt_sRT=[]
            gt_size=[]
            gt_hand=[]
            pred_path=self.pred_path

            for ins_j in range(0,len(filtered_elements)-1):
                sign_gt=ins_j+1
                instance_index=torch.where(instance_labels==filtered_elements[sign_gt])
                obsv_pcd=all_obsv_pcd[instance_index]
                obsv_npcs=gt_npcs[instance_index]
                npcs_max=obsv_npcs.max(0)[0]                                            #最大最小值
                npcs_min=obsv_npcs.min(0)[0]

                scale, rotation, translation,sRT_tmp ,filed_idx = umeyama.estimateSimilarityTransform(obsv_npcs, obsv_pcd, False)
                gt_class_ids.append(sem_labels[instance_index[0][0]].cpu())
                gt_sRT.append(sRT_tmp.cpu())
                gt_size.append(((npcs_max-npcs_min)*scale).cpu())
                gt_hand.append(1)
                s4 = torch.pow(torch.linalg.det(gt_sRT[-1][:3, :3]), 1/3)
                gt_sRT[-1][:3, :3] = gt_sRT[-1][:3, :3] / s4
                exists_or_mkdir(pred_path)

            result = {}
            result['file_name']=data['file_name'][0]
            result['gt_class_ids'] = gt_class_ids
            result['gt_RTs'] = gt_sRT
            result['gt_scales'] = gt_size
            result['gt_handle_visibility'] = gt_hand

            classes_rgbd=classes_rgbd.cuda()

            pcld = all_obsv_pcd

            nocs_tmp=pred_nocs[0].transpose(1,0).reshape(-1,10,3,cfg_bins['ce_loss_bins']).cuda()

            sem_indices=classes_rgbd.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(-1, 1, 3, cfg_bins['ce_loss_bins']).cuda()

            selected_pred = torch.gather(nocs_tmp, 1, sem_indices).squeeze(1)
            _, max_indices = torch.max(selected_pred, dim=2)



            labels_array = classes_rgbd
            if torch.all(classes_rgbd == 0):
                return
            
            pred_inst_label=[]
            pred_class_ids=[]
            pred_scores=[]
            pred_sRT=[]
            pred_size=[]

            for j in range(1,num_classes):

                cate_mask=(labels_array==j)

                true_indices = torch.nonzero(cate_mask, as_tuple=False).flatten()
                cate_xyz=pcld[cate_mask]
                max_indices_sem=max_indices[cate_mask]
                if j==3:
                    mean_shift=ms.MeanShiftTorch(bandwidth=0.02,max_iter=10)
                elif j==5:
                    mean_shift=ms.MeanShiftTorch(bandwidth=0.07,max_iter=10)
                else:
                    mean_shift=ms.MeanShiftTorch(bandwidth=0.1,max_iter=6)
                tmp = cate_xyz + pred_offset[0].transpose(1, 0)[cate_mask]

                if tmp is None or tmp.numel() == 0:
                    continue

                ctr_lst, lb, n_in_lst = mean_shift.fit_multi_clus(cate_xyz + pred_offset[0].transpose(1, 0)[cate_mask])
                if j==3:    
                    instance_len=np.where(np.array(n_in_lst)>30)[0].shape[0]    #1-instance_len
                elif j==2:
                    instance_len=np.where(np.array(n_in_lst)>20)[0].shape[0]
                elif j==1:
                    instance_len=np.where(np.array(n_in_lst)>20)[0].shape[0]
                elif j==5:
                    instance_len=np.where(np.array(n_in_lst)>100)[0].shape[0]
                elif j==8:
                    instance_len=np.where(np.array(n_in_lst)>20)[0].shape[0]
                else:
                    instance_len=np.where(np.array(n_in_lst)>200)[0].shape[0]
                mask=lb<=instance_len
                lb_mask=lb[mask]

                true_indices_mask=true_indices[mask.cpu()]

                max_indices_sem_cluster=max_indices_sem[mask.cpu()]
                for i in range(1,instance_len+1):       #range(1,instance_len+1)
                    idx = torch.where(lb_mask == i)[0]
                    true_indices_mask_instance=true_indices_mask[idx]
                    pred_inst_onehot = torch.zeros(data['pts'].shape[1], dtype=torch.int, device='cuda')
                    pred_inst_onehot[true_indices_mask_instance]=1
                    part_npcs=((max_indices_sem_cluster[idx]-(cfg_bins['ce_loss_bins']-1)/2)/(cfg_bins['ce_loss_bins']-1))
                    part_obsv_pcd=cate_xyz[mask][idx]

                    ume_result = umeyama.estimateSimilarityTransform(part_npcs, part_obsv_pcd, False)
                    if ume_result is None:
                        continue
                    elif len(ume_result) < 5:
                        continue

                    else:
                        scale2, rotation2, translation2, sRT_tmp, pred_filed_idx = ume_result
                    try:
                        if sRT_tmp is None:
                            continue

                        part_npcs_fil1=part_npcs[pred_filed_idx]
                        part_obsv_pcd_fil1=part_obsv_pcd[pred_filed_idx]
                    except TypeError as e:
                        print(f"Caught an exception: {e}")
                        return

                    s4 = torch.pow(torch.linalg.det(sRT_tmp[:3, :3]), 1/3)
                    sRT_tmp[:3, :3]=sRT_tmp[:3, :3] / s4

                    trans_npcs=transform_coordinates_3d(part_npcs_fil1.cpu().numpy().transpose(1,0)*scale2.cpu().numpy(),sRT_tmp.cpu().numpy())

                    part_obsv_pcd_np = part_obsv_pcd_fil1.cpu().numpy()
                    trans_npcs_np = trans_npcs.transpose(1, 0)
                    distances = np.linalg.norm(part_obsv_pcd_np - trans_npcs_np, axis=1)
                    mean_distance = np.mean(distances)
                    distance_threshold = mean_distance * 1.2 
                    filtered_indices = distances < distance_threshold
                    if np.sum(filtered_indices)<17:
                        continue
                    pred_sRT.append(sRT_tmp)
                    pred_size.append(2*torch.abs(part_npcs_fil1[filtered_indices]).max(0)[0]*scale2)
                    pred_class_ids.append(j)
                    pred_scores.append(len(idx))
                    pred_inst_label.append(pred_inst_onehot)

            if load_from_pkl=="":
                result['pred_class_ids'] = pred_class_ids
                result['pred_scores'] = pred_scores
                result['pred_RTs'] = pred_sRT
                result['pred_scales'] = pred_size
                result['pred_inst_label']=pred_inst_label
            else:
                with open(load_from_pkl, "rb") as f:
                    load_data_ = pickle.load(f)
                    import pdb
                    pdb.set_trace()
                    pred_class_ids = load_data_['pred_class_ids']
                    pred_scores = load_data_['pred_scores']
                    pred_sRT = load_data_['pred_RTs']
                    pred_size = load_data_['pred_scales']
                    pred_inst_label=load_data_['pred_inst_label']

                    result['pred_class_ids'] = pred_class_ids
                    result['pred_scores'] = pred_scores
                    result['pred_RTs'] = pred_sRT
                    result['pred_scales'] = pred_size
                    result['pred_inst_label']=pred_inst_label



            if not os.path.exists(pred_path):
                os.makedirs(pred_path)
            save_path = os.path.join(pred_path, 'results_{}.pkl'.format(data['file_name'][0]))
            with open(save_path, 'wb') as f:
                cPickle.dump(result, f)

           
            trans = np.loadtxt(load_path+'scale/'+f"{data['file_name'][0]}"+".txt")
            if (pred_sRT is None):
                return
            try:
                numpy_sRT = [tensor.cpu().numpy() for tensor in pred_sRT if tensor is not None]
                numpy_size = [tensor.cpu().numpy() for tensor in pred_size if tensor is not None]

            except AttributeError as e:
                print("Error converting pred_sRT elements to numpy arrays.")
                print(f"Exception: {e}")
                print("Ensure all elements in pred_sRT are tensors.")
                import pdb
                pdb.set_trace()
            if not numpy_sRT:
                return
            numpy_sRT = np.stack(numpy_sRT)
            numpy_size = np.stack(numpy_size)
            numpy_size=numpy_size*trans[0]
            numpy_sRT[:,0:3,3]=numpy_sRT[:,0:3,3]* trans[0]+trans[1:4]
            metafile = load_meta(load_path, data['file_name'][0])

            if len(gt_sRT) ==0:
                return 0
            gt_sRT = np.stack(gt_sRT)
            gt_size = np.stack(gt_size)
            gt_size=gt_size*trans[0]

            gt_sRT[:,0:3,3]=gt_sRT[:,0:3,3]* trans[0]+trans[1:4]
            metafile = load_meta(load_path, data['file_name'][0])
            if visual:
                intrinsics = metafile['camera_intrinsic']
                original_shape = (3, 3)
                mat_K=np.array(intrinsics).reshape(original_shape)
                image_dir=os.path.join(pred_path, "image_visual")
                os.makedirs(image_dir, exist_ok=True)
                draw_detections_pcs_pred(rgb_image,image_dir, f"{data['file_name'][0]}", 1, mat_K, np.array(numpy_sRT), np.array(numpy_size),draw_gt=True,color=(255,255,0))
                draw_detections_pcs_pred(rgb_image,image_dir, f"{data['file_name'][0]}_gt", 1, mat_K, np.array(gt_sRT), \
                                        np.array(gt_size),draw_gt=True,color=(0,255,0),joint=True)
        return 0



    def eval_func_yoeo(self,data):
        self.is_testing = True
        self.net.eval()
        with torch.no_grad():
            data['pts_feat'] = self.net(data, mode='pts_feature')
            self.evaluate(data, load_from_pkl='', visual=self.cfg.visual)
        return 0
