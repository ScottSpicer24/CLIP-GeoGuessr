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
        self.device = device
        
        self.probe1 = nn.Linear() #TODO

    def forward(self, data):
        # Initialize varaibles
        model = self.clip
        preprocess = self.pp
        device = self.device
        
        # Pull and preprocess the image and the text description from the data
        images = [preprocess(item[0]).to(device) for item in data]
        texts = [item[1] for item in data]  # Extract the text as a list of strings
        text_batch = clip.tokenize(texts).to(device) # returns a batch tensor when you pass it a list of strings. There’s no need to stack this output again.

        # Create a batch by stacking the preprocessed images
        image_batch = torch.stack(images).to(device)
        
        # Get the image and text embeddings
        image_embed = model.encode_image(image_batch)
        text_embed = model.encode_text(text_batch)

        
        
        return 0   