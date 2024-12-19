# stream the dataset and get the labels
import torch
from PIL import Image
from datasets import load_dataset

dataset = load_dataset('osv5m/osv5m', full=True, split='train', streaming=True, trust_remote_code=True) # Stream the data due to the size

for i, example in enumerate(dataset):
    print(f"Example {i+1}:")
    print(example)
    image = example['image']
    image.show()

    #TODO
    # print labels (country, continent, city...) to a csv