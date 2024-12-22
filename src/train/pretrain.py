import torch
import torch.optim as optim
import clip
from PIL import Image
from datasets import load_dataset
import argparse
from InfoNCELoss import InfoNCELoss

'''
Further pre-trains the CLIP model for geolocalization

JUST the 2 encoders NOT the linear probe
'''

def main(FLAGS):
    # Pull parameters from FLAGS
    lr = FLAGS.learning_rate
    num_epochs = FLAGS.num_epochs
    batch_size = FLAGS.batch_size
    
    # Check if cuda is available
    use_cuda = torch.cuda.is_available()
    # Set proper device based on cuda availability 
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Device selected: {device}")
    
    # Initialize the model and send to device 
    model, preprocess = clip.load("ViT-B/32", device=device)
    # Set the loss function 
    criterion = InfoNCELoss()
    # Set the optimizer function
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    # Load the dataset
    dataset = load_dataset('osv5m/osv5m', full=True, split='train', streaming=True, trust_remote_code=True) # Stream the data due to the size
    
    #  For the entire training dataset an epoch amount of times
    for epoch in range(num_epochs):
        print("Epoch ", epoch+1)
        # Pull out a batch size group of them
        batch = []
        for i, item in enumerate(dataset):
            if i == 34:
                break
            # Extract and save the image and it's metadata
            image = item['image'] # preprocess(Image.open(item['image'])).to(device)
            text = gen_synth_text(item) # clip.tokenize(gen_synth_text(item)).to(device)
            batch.append((image, text))

            #print(text)
            
            # Train the batch then clear it when you are done
            if len(batch) == batch_size:
                train_batch(model, preprocess, batch, criterion, optimizer, device)
                batch.clear()

            # Train leftovers at end of dataset
            if len(batch) > 0:
                train_batch(model, preprocess, batch, criterion, optimizer, device)
                batch.clear()


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

def train_batch(model, preprocess, data, criterion, optimizer, device):
    # Set the model to be trained and zero gradients 
    model.train()
    optimizer.zero_grad()
    
    # Pull and preprocess the image and the text description from the data
    images = [preprocess(Image.open(item[0])).to(device) for item in data]
    texts = [clip.tokenize(item[1]).to(device) for item in data]

    # Create a batch by stacking the preprocessed images and text
    image_batch = torch.stack(images)
    text_batch = torch.stack(texts)
    
    # Get the image and text embeddings
    image_embed = model.encode_image(image_batch)
    text_embed = model.encode_text(text_batch)

    # Compute the contrastive loss and optimize model from loss
    loss = criterion(image_embed, text_embed)
    loss.backward()
    optimizer.step()

    print(loss)


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
    