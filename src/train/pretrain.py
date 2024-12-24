import torch
import torch.optim as optim
import clip
from PIL import Image
from datasets import load_dataset
import argparse
from InfoNCELoss import InfoNCELoss
import os
import csv

'''
Further pre-trains the CLIP model for geolocalization

JUST the 2 encoders NOT the linear probe
'''

def main(FLAGS):
    # Pull parameters from FLAGS
    lr = FLAGS.learning_rate
    num_epochs = FLAGS.num_epochs
    batch_size = FLAGS.batch_size
    print(f"LR selected: {lr}, epochs: {num_epochs}, batch size: {batch_size}")
    
    # Check if cuda is available
    use_cuda = torch.cuda.is_available()
    # Set proper device based on cuda availability 
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Device selected: {device}")
    
    # Initialize the model and send to device 
    model, preprocess = clip.load("ViT-B/32", device=device)
    # Set the loss function 
    criterion = InfoNCELoss().to(device)
    # Set the optimizer function
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    # Load the dataset
    dataset = load_dataset('osv5m/osv5m', full=True, split='train', streaming=True) # Stream the data due to the size
    
    #  For the entire training dataset an epoch amount of times
    for epoch in range(num_epochs):
        print("Epoch ", epoch+1)
        # Pull out a batch size group of them
        batch = []
        for i, item in enumerate(dataset):
            if i%1000 == 0:
                string = f"Epoch: {epoch+1}, i:{i}"
                csvPrint("CLIP_pretrain_trial.csv", string)
            
            # Extract and save the image and it's metadata
            image = item['image'] # preprocess(Image.open(item['image'])).to(device)
            text = gen_synth_text(item) # clip.tokenize(gen_synth_text(item)).to(device)
            batch.append((image, text))
            
            # Train the batch then clear it when you are done
            if len(batch) == batch_size:
                train_batch(model, preprocess, batch, criterion, optimizer, device)
                batch.clear()

            # Train leftovers at end of dataset
            if len(batch) > 0:
                train_batch(model, preprocess, batch, criterion, optimizer, device)
                batch.clear()
        
        # After the last epoch save model
        torch.save(model.state_dict(), 'clip_geolocalization_weights.pth')
        print("Model weights saved to clip_geolocalization_weights.pth")



# Generates synthetic text for the language encoder
def gen_synth_text(data):
    # Extract attribute values 
    country = data['country'].strip()   
    region = data['region'].strip()
    sub_region = data['sub-region'].strip()
    city = data['city'].strip()

    # Generate and return string
    string = f"A street view image in the country of {country}, within the region of {region}, more specifically {sub_region}, near the town or city of {city}."
    return string

def csvPrint(path, item):
    # See if path exist, and append to it
    csv_exists = os.path.exists(path)
    attr = path.rstrip(".csv") # For the header
    with open(path, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write header if file doesn't exist
        if not csv_exists:
            writer.writerow([attr])
        
        # Write attribute values to csv
        writer.writerow([item])

def train_batch(model, preprocess, data, criterion, optimizer, device):
    # Set the model to be trained and zero gradients 
    model.train()
    optimizer.zero_grad()
    
    # Pull and preprocess the image and the text description from the data
    images = [preprocess(item[0]).to(device) for item in data]
    texts = [item[1] for item in data]  # Extract the text as a list of strings
    text_batch = clip.tokenize(texts).to(device) # returns a batch tensor when you pass it a list of strings. There’s no need to stack this output again.

    # Create a batch by stacking the preprocessed images
    image_batch = torch.stack(images).to(device)
    
    # Get the image and text embeddings
    image_embed = model.encode_image(image_batch)
    text_embed = model.encode_text(text_batch)

    # Compute the contrastive loss and optimize model from loss
    loss = criterion(image_embed, text_embed)
    loss.backward()
    optimizer.step()

    #print(loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('CLIP-GG arguments.')
    
    parser.add_argument('--learning_rate',
                        type=float, default=0.000001,
                        help='Initial learning rate.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=3,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--batch_size',
                        type=int, default=32,
                        help='Batch size. Must divide evenly into the dataset sizes.')
    
    FLAGS = None
    FLAGS, unparsed = parser.parse_known_args()
    
    main(FLAGS)
    