import torch
import clip
from PIL import Image
from datasets import load_dataset

'''
Used to mess around and figure things out
'''

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

dataset = load_dataset('osv5m/osv5m', full=True, split='train', streaming=True, trust_remote_code=True)
for i, example in enumerate(dataset):
    if i == 5:
        break
    print(f"Example {i+1}:")
    print(example)


'''continents = ['Africa', 'Antartica', 'Asia', 'Europe', 'North America', 'Oceania', 'South America']
text_inputs = torch.cat([clip.tokenize(f"A street view photo taken in the continent of {c}") for c in continents]).to(device)

for i, example in enumerate(dataset):
    if i == 5:
        break
    print(f"Example {i}:")
    print(example)
    image = example['image'] # already a PIL image instance, so do not Image.open() it.
    image.show()
    print(" ")
    
    image_input = preprocess(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)
        
        logits_per_image, logits_per_text = model(image_input, text_inputs)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    print("Label probs:", probs)
    print("Actual country:", example['country'])
'''

