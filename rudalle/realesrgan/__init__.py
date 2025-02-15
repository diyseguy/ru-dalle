# -*- coding: utf-8 -*-
import os

from huggingface_hub import hf_hub_url, hf_hub_download

from .model import RealESRGAN


MODELS = {
    'x2': dict(
        scale=2,
        repo_id='sberbank-ai/Real-ESRGAN',
        filename='RealESRGAN_x2.pth',
    ),
    'x4': dict(
        scale=4,
        repo_id='sberbank-ai/Real-ESRGAN',
        filename='RealESRGAN_x4.pth',
    ),
    'x8': dict(
        scale=8,
        repo_id='sberbank-ai/Real-ESRGAN',
        filename='RealESRGAN_x8.pth',
    ),
}


def get_realesrgan(name, device='cpu', fp16=False, cache_dir='/tmp/rudalle'):
    assert name in MODELS
    config = MODELS[name]
    model = RealESRGAN(device, config['scale'], fp16=fp16)
    cache_dir = os.path.join(cache_dir, name)
    repo_id = config['repo_id']
    repo_filename = config['filename']
    config_file_url = hf_hub_url(repo_id=config['repo_id'], filename=config['filename'])
    #'https://huggingface.co/sberbank-ai/Real-ESRGAN/resolve/main/RealESRGAN_x2.pth'
    try:
        hf_hub_download(repo_id, cache_dir=cache_dir, filename=repo_filename)
    except Exception as ex:
        print(ex)
        raise ex
    model.load_weights(os.path.join(cache_dir, config['filename']))
    print(f'{name} --> ready')
    return model
