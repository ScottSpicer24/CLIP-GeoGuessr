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

    # Pull out a batch size 





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
    