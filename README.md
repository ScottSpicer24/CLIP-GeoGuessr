# CLIP-GeoGuessr
Fine Tuned version of OpenAI's CLIP model for Image Geolocation. Git repo: https://github.com/openai/CLIP.git 

This is in the conda enviorment but you also can obtain the model here:
```bash
pip install git+https://github.com/openai/CLIP.git 
```

## Setup Instructions

### 1. Clone the Repository
First, clone the repository:

```bash
git clone https://github.com/ScottSpicer24/CLIP-GeoGuessr.git
```

### 2. Setup the Conda Environment & Install Dependencies
Create and activate a new conda environment called `CLIP-GG`:

```bash
conda create -n CLIP-GG python=3.9
conda activate CLIP-GG
```

Recreate the environment to install the required Python libraries:

```bash
conda env create -f environment.yml
# pip install git+https://github.com/openai/CLIP.git 
```

### 3. Obtain Data
I use the OpenStreetView-5M dataset. The dataset is 5 million images and a csv of a metadata for all of them, so it's big
Dataset card: https://huggingface.co/datasets/osv5m/osv5m

#### Option 1:
Directly load the dataset using load_dataset:
```python
from datasets import load_dataset
dataset = load_dataset('osv5m/osv5m', full=False)
```

#### Option 2:
Manual download part of the dataset from huggingface:
Image zips: https://huggingface.co/datasets/osv5m/osv5m/tree/main/images/train
CSV metadata: https://huggingface.co/datasets/osv5m/osv5m/tree/main
Then run unzip.py in ./src/preprocessing

#### Option 3:
Download the full dataset:
```python
# download the full dataset
from huggingface_hub import snapshot_download
import os
import zipfile

snapshot_download(repo_id="osv5m/osv5m", local_dir="datasets/osv5m", repo_type='dataset')
for root, dirs, files in os.walk("datasets/osv5m"):
    for file in files:
        if file.endswith(".zip"):
            with zipfile.ZipFile(os.path.join(root, file), 'r') as zip_ref:
                zip_ref.extractall(root)
                os.remove(os.path.join(root, file))


```
