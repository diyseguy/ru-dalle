# -*- coding: utf-8 -*-
import os

import torch
from huggingface_hub import hf_hub_url, hf_hub_download


MODELS = {
    'unet_effnetb5': dict(
        encoder_name='efficientnet-b5',
        repo_id='sberbank-ai/rudalle-Emojich',
        filename='pytorch_model_v2.bin',
        classes=2,
    ),
}


def get_emojich_unet(name, cache_dir='/tmp/rudalle'):
    assert name in MODELS
    config = MODELS[name]
    try:
        import segmentation_models_pytorch as smp
    except ImportError:
        import logging
        logging.warning('If you would like to use emojich_unet, you should reinstall timm package:'
                        '"pip install timm==0.4.12"')
        return
    model = smp.Unet(
        encoder_name=config['encoder_name'],
        encoder_weights=None,
        in_channels=3,
        classes=config['classes'],
    )
    cache_dir = os.path.join(cache_dir, name)
    repo_filename = config['filename']
    repo_id = config['repo_id']
    config_file_url = hf_hub_url(repo_id=repo_id, filename=f'{name}/{filename}')
    # 'https://huggingface.co/sberbank-ai/rudalle-Emojich/resolve/main/unet_effnetb5/pytorch_model_v2.bin'
    try:
        hf_hub_download(repo_id, cache_dir=cache_dir, filename=repo_filename)
    except Exception as ex:
        print(ex)
        raise ex
    checkpoint = torch.load(os.path.join(cache_dir, config['filename']), map_location='cpu')
    model.load_state_dict(checkpoint)
    print(f'{name} --> ready')
    return model
