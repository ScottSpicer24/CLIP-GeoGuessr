import torch
import torch.optim as optim
import clip
from PIL import Image
from datasets import load_dataset
import argparse
from InfoNCELoss import InfoNCELoss

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


def train(model, data, optimizer, criterion, device, epoch):
    '''
    moddel: The CLIP model that is being run
    '''

    
    model.train()

    total_loss = 0
    correct = 0
    total_samples = 0

     # go through the Dataloader and train the samples (this is one epoch)
    for index, data_sample in enumerate(data):
        # extract the feature values and the ground truth labels 
        sample_features, target_label = data_sample
        #print(f"TGT: {target_label}")

        # Push data/label to correct device
        sample_features, target_label = sample_features.to(device), target_label.to(device)

        # Reset optimizer gradients.
        optimizer.zero_grad()

        # have the model make predictions
        prediction_label = model(sample_features)

        # Compute loss based on criterion (CrossEntropy in this case)
        loss = criterion(prediction_label, target_label)
        # Computes gradient based on final loss
        loss.backward()
        # Store loss
        total_loss += loss.item()
        
        # Optimize model parameters based on learning rate and gradient 
        # The model and the SGD optimizer were connected in the main() function 
        optimizer.step()

        # Get predictions and compute accuracy
        _, predicted = torch.max(prediction_label, 1)
        correct += (predicted == target_label).sum().item()
        total_samples += target_label.size(0)

    # Calculate average loss and accuracy
    avg_loss = total_loss / len(data)
    accuracy = correct / total_samples * 100

    # Print loss and accuracy for the epoch
    print("Accuracy(as %) = ", accuracy)
    print("Avg Loss = ", avg_loss)
    print(" ")

def test(model, device, test_loader, criterion):
    '''
    Tests the model.
    model: The model to test. Should already be in correct device.
    device: 'cuda' or 'cpu'.
    test_loader: dataloader for test samples.
    '''
    
    # Set model to eval mode to notify all layers.
    model.eval()
    
    losses = []
    all_predictions = []
    all_labels = []
    correct = 0
    
    # Set torch.no_grad() to disable gradient computation and backpropagation
    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            data, target = sample
            data, target = data.to(device), target.to(device)
            
            # Predict for data by doing forward pass
            output = model(data)

            pred = output.argmax(dim=1)
            all_predictions.append(pred)
            all_labels.append(target)
            
            # ======================================================================
            # Compute loss based on same criterion as training
            loss = criterion(output, target)
            
            # Append loss to overall test loss
            losses.append(loss.item())
            
            # Get predicted index by selecting maximum log-probability
            pred = output.argmax(dim=1, keepdim=True)
            
            # ======================================================================
            # Count correct predictions overall 
            correct += pred.eq(target.view_as(pred)).sum().item()

    # move predictions and lables to cpu and numpy for MSE Calculation
    all_predictions = torch.cat(all_predictions).cpu().numpy()
    all_labels = torch.cat(all_labels).cpu().numpy()
    # Compute Mean Squared Error
    mse = mean_squared_error(all_labels, all_predictions)
    print(f"Mean Squared Error (MSE): {mse:.4f}")

    #test_loss = float(np.mean(losses))
    accuracy = 100. * correct / len(test_loader.dataset)

    print('Test set accuracy: {}/{} ({:.0f}%)'.format(
        correct, len(test_loader.dataset), accuracy))

def main(FLAGS):
    # Pull parameters from FLAGS
    lr = FLAGS.learning_rate
    num_epochs = FLAGS.num_epochs
    batch_size = FLAGS.batch_size
    
    # Check if cuda is available
    use_cuda = torch.cuda.is_available()
    
    # Set proper device based on cuda availability 
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # Initialize the model and send to device 

    # Set the loss function 
    criterion = InfoNCELoss()
    # Set the optimizer function
    optimizer = optim.AdamW(model.parameters(), lr=FLAGS.learning_rate)

    


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

