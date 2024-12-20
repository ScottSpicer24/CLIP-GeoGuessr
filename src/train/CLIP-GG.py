import clip
import torch
import torch.nn as nn
import torch.functional as F


class CLIP_GG(nn.Module):
    def __init__(self, device):
        super(CLIP_GG, self).__init__()
        model, preprocess = clip.load("ViT-B/32", device=device)
        self.clip = model
        self.pp = preprocess
        
        self.probe1 = nn.Linear() #TODO

    def forward(self, image, text):
        #
        model = self.clip
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        
        return 0