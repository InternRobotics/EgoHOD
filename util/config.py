
import os
import os.path as osp

from omegaconf import OmegaConf


def load_config(cfg_file):
    cfg = OmegaConf.load(cfg_file)
    if '_base_' in cfg:
        if isinstance(cfg._base_, str):
            base_cfg = OmegaConf.load(osp.join(osp.dirname(cfg_file), cfg._base_))
        else:
            base_cfg = OmegaConf.merge(OmegaConf.load(f) for f in cfg._base_)
        cfg = OmegaConf.merge(base_cfg, cfg)
    return cfg

def get_config(args):
    cfg = load_config(args.config_file)
    OmegaConf.set_struct(cfg, True)
    OmegaConf.set_readonly(cfg, True)

    return cfg
