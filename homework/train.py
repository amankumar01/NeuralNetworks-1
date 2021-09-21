from .models import ClassificationLoss, model_factory, save_model
from .utils import accuracy, load_data
import torch
import torch.utils.tensorboard as tb
import time
from tqdm.notebook import tqdm

def train(args):
    model = model_factory[args.model]()
    lr=0.001
    epochs=10
    data_train = load_data("data/train")
    data_val = load_data("data/valid")
    """
    Your code here

    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Setup the loss function to use
    loss_function = torch.nn.CrossEntropyLoss()

    # Setup the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Wrap in a progress bar.
    for epoch in tqdm(range(epochs)):
        # Set the model to training mode.
        model.train()

        for x, y in data_train:
            x = x.to(device)
            y = y.to(device)

            # Forward pass through the network
            output = model(x)

            # Compute loss
            loss = loss_function(output, y)
            
            # update model weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        # Set the model to eval mode and compute accuracy.
        # No need to change this, but feel free to implement additional logging.
        model.eval()

        accuracys_val = list()

        for x, y in data_val:
            x = x.to(device)
            y = y.to(device)

            y_pred = model.predict(x)
            accuracy_val = (y_pred == y).float().mean().item()
            accuracys_val.append(accuracy_val)

        accuracy = torch.FloatTensor(accuracys_val).mean().item()

    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', choices=['linear', 'mlp'], default='linear')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
