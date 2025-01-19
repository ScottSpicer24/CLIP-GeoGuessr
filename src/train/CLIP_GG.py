import clip
import torch
import torch.nn as nn
import torch.functional as F
import csv



class CLIP_GG(nn.Module):
    def __init__(self, device):
        super(CLIP_GG, self).__init__()
        model, preprocess = clip.load("ViT-B/32", device=device)
        self.clip = model
        self.pp = preprocess
        self.device = device

        self.geolabels = intialize_geolabels()
        self.countries = [country for country in self.geolabels.keys()]
        
        self.probe1 = nn.Linear(self.clip.visual.output_dim, len(self.geolabels)) 

        # city-level probes for each country
        self.probe2 = nn.ModuleDict({
            country : nn.Linear(self.clip.visual.output_dim, len(cities)) for country, cities in self.geolabels.items()
        })
        

    def forward(self, data):
        # Initialize varaibles
        model = self.clip
        preprocess = self.pp
        device = self.device
        
        # Pull and preprocess the image and the text description from the data
        images = [preprocess(item[0]).to(device) for item in data]
        # texts = [item[1] for item in data]  # Extract the text as a list of strings
        # text_batch = clip.tokenize(texts).to(device) # returns a batch tensor when you pass it a list of strings. There’s no need to stack this output again.

        # Create a batch by stacking the preprocessed images
        image_batch = torch.stack(images).to(device)
        
        # Get the image embeddings
        # We don't need the text embeddings as that is what is being predected
        image_embed = model.encode_image(image_batch)
        
        # Pass to emnbeddings into probe
        logits1 = self.probe1(image_embed)
        pred_country = self.countries[torch.argmax(logits1, dim=1)]
        
        city_probe = self.probe2[pred_country]
        logits2 = city_probe(image_embed)
        city_list = self.geolabels[pred_country]
        pred_city = city_list[torch.argmax(logits2, dim=1)]

        return pred_country, pred_city  
    

def intialize_geolabels():
    path = "../preprocess/geolabels.csv"
    res = {}
    with open(path, mode='r') as file:
        csvFile = csv.reader(file)
        for line in csvFile:
            country = line[0]
            res[country] = line[1]
    
    return res
