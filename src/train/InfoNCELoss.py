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
        print(f'Embedings:\nImage: {image_embeds},\nText: {text_embeds}\n')
        n = image_embeds.size(0)

        # Cosine Similarities
        '''
        From original paper regarding temperture:
        "clipped to prevent scaling the logits by more than 100 which we found necessary to prevent training instability"
        scaling factor = e^t --> e^t < 100 --> e^t == 100 when t = ln(100), so cap at ln(100) or 4.605..
        '''
        print("temp: ", self.temperature)
        logits = torch.matmul(image_embeds, text_embeds.T) * torch.exp(self.temperature.clamp(max=4.6))
        #print("logits ", logits)

        # Symetric loss between the 2 different embeddings
        labels = torch.arange(n).to(image_embeds.device) # each index is it's own label
        loss_image_to_text = self.criterion(logits, labels)
        #print("img to text ", loss_image_to_text)
        loss_text_to_image = self.criterion(logits.T, labels)
        #print("txt to img ", loss_text_to_image)

        # Return results 
        loss = (loss_image_to_text + loss_text_to_image) / 2
        return loss


