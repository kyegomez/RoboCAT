import os
import sys
import yaml

import torch
import requests
import numpy as np

from PIL import Image
from omegaconf import OmegaConf
from taming.models.vqgan import VQModel, GumbelVQ

import torchvision.transforms as T
import torchvision.transforms.functional as TF

from io import BytesIO

class VQGAN_F8_8192:
    def __init__(self, device):
        self.device = device
        self.download_vqgan_gumbel_f8()
        self.config = self.load_config("logs/vqgan_gumbel_f8/configs/model.yaml")
        self.model = self.load_vqgan(self.config, ckpt_path="logs/vqgan_gumbel_f8/checkpoints/last.ckpt", is_gumbel=True).to(self.device)

    def download_vqgan_gumbel_f8(self):
        os.makedirs("logs/vqgan_gumbel_f8/checkpoints", exist_ok=True)
        os.makedirs("logs/vqgan_gumbel_f8/configs", exist_ok=True)
        ckpt_url = "https://heibox.uni-heidelberg.de/f/34a747d5765840b5a99d/?dl=1"
        config_url = "https://heibox.uni-heidelberg.de/f/b24d14998a8d4f19a34f/?dl=1"
        ckpt_path = "logs/vqgan_gumbel_f8/checkpoints/last.ckpt"
        config_path = "logs/vqgan_gumbel_f8/configs/model.yaml"
        if not os.path.exists(ckpt_path):
            with requests.get(ckpt_url, stream=True) as r:
                r.raise_for_status()
                with open(ckpt_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
        if not os.path.exists(config_path):
            with requests.get(config_url, stream=True) as r:
                r.raise_for_status()
                with open(config_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

    def load_config(self, config_path, display=False):
        config = OmegaConf.load(config_path)
        if display:
            print(yaml.dump(OmegaConf.to_container(config)))
        return config

    def load_vqgan(self, config, ckpt_path=None, is_gumbel=False):
        if is_gumbel:
            model = GumbelVQ(**config.model.params)
        else:
            model = VQModel(**config.model.params)
        if ckpt_path is not None:
            sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
            missing, unexpected = model.load_state_dict(sd, strict=False)
        return model.eval()

    def preprocess(self, img, target_image_size=256):
        s = min(img.size)

        if s < target_image_size:
            raise ValueError(f'min dim for image {s} < {target_image_size}')

        r = target_image_size / s
        s = (round(r * img.size[1]), round(r * img.size[0]))
        img = TF.resize(img, s, interpolation=PIL.Image.LANCZOS)
        img = TF.center_crop(img, output_size=2 * [target_image_size])
        img = torch.unsqueeze(T.ToTensor()(img), 0)
        return img
    
    def collect_and_preprocess_data(self, data_urls):
        preprocessed_data = []
        for url in data_urls:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
            preprocessed_img = self.preprocess(img)
            preprocessed_data.append(preprocessed_img)
        return preprocessed_data
    

    def reconstruct(self, img):
        x = self.preprocess(img)
        x = x.to(self.device)
        z, _, [_, _, indices] = self.model.encode(x)
        xrec = self.model.decode(z)
        return xrec
    

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
vqgan = VQGAN_F8_8192(device)


#downlaod and image from the internet
image_url = "https://images.unsplash.com/photo-1592194996308-7b43878e84a6"
response = requests.get(image_url)
input_image = Image.open(BytesIO(response.content))


#reconstruct the image using the VQGAN Model
reconstructed_image = vqgan.reconstruct(input_image)

#display the input and recontructed images
input_image.show()
reconstructed_image.show()



# Collect proprioceptive observations, agent actions, and images
data_urls = [
    "https://images.unsplash.com/photo-1592194996308-7b43878e84a6",
    "https://images.unsplash.com/photo-1582719508461-905c673771fd",
    # ...
]

#preprocess the collected data for tokenization
preprocessed_data = vqgan.collect_and_preprocess_data(data_urls)