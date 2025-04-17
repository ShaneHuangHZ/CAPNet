import argparse
from ipdb import set_trace

def get_config():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--local-rank", type=int, default=0, help="Local rank for distributed training")
    parser.add_argument("--pred_path", type=str, default='./result_cfg', help="Local rank for distributed training")
    """ dataset """
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--batch_size', type=int, default=192)
    parser.add_argument('--max_batch_size', type=int, default=192)
    parser.add_argument('--pose_mode', type=str, default='rot_matrix')

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--percentage_data_for_train', type=float, default=1.0) 
    parser.add_argument('--percentage_data_for_val', type=float, default=1.0) 
    parser.add_argument('--percentage_data_for_test', type=float, default=1.0) 
    parser.add_argument('--train_source', type=str)
    parser.add_argument('--val_source', type=str)
    parser.add_argument('--test_source', type=str)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_points', type=int, default=24576)         
    parser.add_argument('--per_obj', type=str, default='')
    parser.add_argument('--num_workers', type=int, default=32)
    
    
    """ model """
    parser.add_argument('--sampler_mode', nargs='+')
    parser.add_argument('--sampling_steps', type=int)
    parser.add_argument('--sde_mode', type=str, default='ve')
    parser.add_argument('--regression_head', type=str, default='Rx_Ry_and_T')
    parser.add_argument('--pointnet2_params', type=str, default='light')
    parser.add_argument('--pts_encoder', type=str, default='pointnet2') 
    parser.add_argument('--energy_mode', type=str, default='IP') 
    parser.add_argument('--s_theta_mode', type=str, default='score') 
    parser.add_argument('--norm_energy', type=str, default='identical')
    parser.add_argument('--dino', type=str, default='pointwise')
    parser.add_argument('--scale_embedding', type=int, default=180)
    parser.add_argument('--pretrained_score_model_path', type=str)
    parser.add_argument('--log_dir', type=str, default='debug')
    parser.add_argument('--infer_split', type=str, default='intra') 
    parser.add_argument('--use_pretrain', default=False, action='store_true')
    parser.add_argument('--parallel', default=False, action='store_true')
    parser.add_argument('--is_train', default=False, action='store_true')
    parser.add_argument('--visual', default=False, action='store_true')

    
    """ testing """
    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--pred', default=False, action='store_true')
    parser.add_argument('--eval_repeat_num', type=int, default=50)
    parser.add_argument('--real_drop', type=int, default=10) # only keep part of the frames in test set for faster inference
    parser.add_argument('--save_video', default=False, action='store_true')
    parser.add_argument('--T0', type=float, default=1.0)
    
    
    parser.add_argument('--img_size', type=int, default=644, help='cropped image size')
    parser.add_argument('--result_dir', type=str, default='', help='result directory')
    parser.add_argument('--clustering', type=int, default=1, help='use clustering to solve multimodal issue')
    parser.add_argument('--clustering_eps', type=float, default=0.05, 
                        help='hyperparameter in clustering (see runners/evaluation_single.py for details)')
    parser.add_argument('--clustering_minpts', type=float, default=0.1667, 
                        help='hyperparameter in clustering (see runners/evaluation_single.py for details)')
    parser.add_argument('--retain_ratio', type=float, default=0.4, help='how much to retain in outlier removal stage')
    
    
    cfg = parser.parse_args()

    cfg.DYNAMIC_ZOOM_IN_PARAMS = {
        'DZI_PAD_SCALE': 1.15,
        'DZI_TYPE': 'uniform',
        'DZI_SCALE_RATIO': 0.115,
        'DZI_SHIFT_RATIO': 0.115
    }
    # pts aug parameters
    cfg.PTS_AUG_PARAMS = {
        'aug_pc_pro': 0.2,
        'aug_pc_r': 0.2,
        'aug_rt_pro': 0.3,
        'aug_bb_pro': 0.3,
        'aug_bc_pro': 0.3
    }
    # 2D aug parameters
    cfg.DEFORM_2D_PARAMS = {
        'roi_mask_r': 3,
        'roi_mask_pro': 0.5
    }
    if cfg.eval or cfg.pred: # disable augmentation in evaluation
        cfg.DYNAMIC_ZOOM_IN_PARAMS['DZI_TYPE'] = 'none'
        cfg.DEFORM_2D_PARAMS['roi_mask_pro'] = 0

    assert cfg.dino in ['none', 'global', 'pointwise']
    
    return cfg

