import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Refrence: 
Original Google paper: https://arxiv.org/abs/1807.03748v2
CLIP paper (pages 4 and 5): https://arxiv.org/pdf/2103.00020
'''

class InfoNCELoss(nn.Module):
    def __init__(self, initial_temp=0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = nn.Parameter(torch.tensor([initial_temp]).log()) # Temperature as a learnable parameter, initialized with log(1/temp)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, image_embeds, text_embeds):
        # Normalize the embeddings
        image_embeds = F.normalize(image_embeds)
        text_embeds = F.normalize(text_embeds)
        n = image_embeds.size(0)

        # Cosine Similarities
        logits = F.cosine_similarity(image_embeds, text_embeds.T) * torch.exp(self.temperature)

        # Symetric loss between the 2 different embeddings
        labels = torch.arange(n).to(image_embeds.device) # each index is it's own label
        loss_image_to_text = self.criterion(logits, labels)
        loss_text_to_image = self.criterion(logits.T, labels)

        # Return results 
        loss = (loss_image_to_text + loss_text_to_image) / 2
        return loss