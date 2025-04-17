import sys
import os
import torch
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from tqdm import tqdm

from datasets.datasets_rgbdart import get_data_loaders_from_cfg, process_batch
from networks.capnet_agent import CapNet 
from configs.config import get_config
from utils.transforms import *
from cutoop.transform import *
from cutoop.data_types import *
from cutoop.eval_utils import *

def inference(cfg, test_intra_loader, test_inter_loader, capnet_agent):
    capnet_agent.eval()
    torch.cuda.empty_cache()
    if(cfg.infer_split == 'test_inter'):
        pbar = tqdm(test_inter_loader)
    elif(cfg.infer_split == 'test_intra'):
        pbar = tqdm(test_intra_loader)
    for i, batch_sample in enumerate(pbar):     
            
        ''' load data '''
        batch_sample = process_batch(
            batch_sample = batch_sample, 
            device=cfg.device, 
            PTS_AUG_PARAMS=cfg.PTS_AUG_PARAMS, 
        )
        with torch.no_grad():
            capnet_agent.eval_func_yoeo(batch_sample)

        



def main():
    cfg = get_config()
    data_loaders = get_data_loaders_from_cfg(cfg=cfg, data_type=['test_inter','test_intra'])
    test_inter_loader = data_loaders['test_inter_loader']
    test_intra_loader = data_loaders['test_intra_loader']
    capnet_agent = CapNet(cfg)
    capnet_agent.load_ckpt(model_dir=cfg.pretrained_score_model_path, model_path=True, load_model_only=True)
    inference(cfg, test_intra_loader, test_inter_loader, capnet_agent)


if __name__ == '__main__':
    main()

