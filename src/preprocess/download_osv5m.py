# download the full dataset
from huggingface_hub import snapshot_download
import os
import zipfile

# Download the data
snapshot_download(repo_id="osv5m/osv5m", local_dir="../datasets/osv5m", repo_type='dataset')
# Will unzip the data in the 
for root, dirs, files in os.walk("../datasets/osv5m/"):
    for file in files:
        if file.endswith(".zip"):
            with zipfile.ZipFile(os.path.join(root, file), 'r') as zip_ref:
                zip_ref.extractall(root)
                os.remove(os.path.join(root, file))