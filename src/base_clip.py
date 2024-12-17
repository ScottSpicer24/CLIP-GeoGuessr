import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device selected: {device}")
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open("./data/00/00/105996411564031.jpg")).unsqueeze(0).to(device)
text = clip.tokenize(["a road", "Montana", "a bottle"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # prints: Label probs: [[0.994549   0.00375925 0.00169171]]