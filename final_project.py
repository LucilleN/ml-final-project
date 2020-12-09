'''
Names (Please write names in <Last Name, First Name> format):
1. Doe, John
2. Doe, Jane

TODO: Project type

TODO: Report what each member did in this project

'''
import argparse
import torch, torchvision
import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()

# Commandline arguments
parser.add_argument('--train_network',
    action='store_true', help='If set, then trains network')
parser.add_argument('--batch_size',
    type=int, default=4, help='Number of samples per batch')
parser.add_argument('--n_epoch',
    type=int, default=1, help='Number of times to iterate through dataset')
parser.add_argument('--learning_rate',
    type=float, default=1e-8, help='Base learning rate (alpha)')
parser.add_argument('--learning_rate_decay',
    type=float, default=0.50, help='Decay rate for learning rate')
parser.add_argument('--learning_rate_decay_period',
    type=float, default=1, help='Period before decaying learning rate')
parser.add_argument('--momentum',
    type=float, default=0.90, help='Momentum discount rate (beta)')
parser.add_argument('--lambda_weight_decay',
    type=float, default=0.0, help='Lambda used for weight decay')

# TODO: please add additional necessary commandline arguments here


args = parser.parse_args()


class FullyConvolutionalNetwork(torch.nn.Module):
    '''
    Fully convolutional network

    Args:
        Please add any necessary arguments
    '''

    def __init__(self):
        super(FullyConvolutionalNetwork, self).__init__()

        # TODO: Design your neural network using
        # (1) convolutional layers
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        # (2) max pool layers
        # https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.max_pool2d
        # (3) average pool layers
        # https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.avg_pool2d
        # (4) transposed convolutional layers
        # https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html

    def forward(self, x):
        '''
            Args:
                x : torch.Tensor
                    tensor of N x d

            Returns:
                torch.Tensor
                    tensor of n_output
        '''

        # TODO: Implement forward function

        return x

def train(net,
          dataloader,
          n_epoch,
          optimizer,
          learning_rate_decay,
          learning_rate_decay_period):
    '''
    Trains the network using a learning rate scheduler

    Args:
        net : torch.nn.Module
            neural network
        dataloader : torch.utils.data.DataLoader
            # https://pytorch.org/docs/stable/data.html
            dataloader for training data
        n_epoch : int
            number of epochs to train
        optimizer : torch.optim
            https://pytorch.org/docs/stable/optim.html
            optimizer to use for updating weights
        learning_rate_decay : float
            rate of learning rate decay
        learning_rate_decay_period : int
            period to reduce learning rate based on decay e.g. every 2 epoch

        Please add any necessary arguments

    Returns:
        torch.nn.Module : trained network
    '''

    # TODO: Define loss function
    loss_func = None

    for epoch in range(n_epoch):

        # Accumulate total loss for each epoch
        total_loss = 0.0

        # TODO: Decrease learning rate when learning rate decay period is met
        # e.g. decrease learning rate by a factor of decay rate every 2 epoch

        for batch, _ in enumerate(dataloader):

            # TODO: Forward through the network

            # TODO: Clear gradients so we don't accumlate them from previous batches

            # TODO: Compute loss function

            # TODO: Update parameters by backpropagation

            # TODO: Accumulate total loss for the epoch

            pass

        mean_loss = total_loss / float(batch)

        # Log average loss over the epoch
        print('Epoch=%d  Loss: %.3f' % (epoch + 1, mean_loss))

    return net

def evaluate(net, dataloader):
    '''
    Evaluates the network on a dataset

    Args:
        net : torch.nn.Module
            neural network
        dataloader : torch.utils.data.DataLoader
            # https://pytorch.org/docs/stable/data.html
            dataloader for training data

        Please add any necessary arguments
    '''

    # Make sure we do not backpropagate
    with torch.no_grad():

        for _ in dataloader:

            # TODO: Forward through the network

            # TODO: Compute evaluation metric(s) for each sample

            pass

    # TODO: Compute mean evaluation metric(s)

    # TODO: Print scores

    # TODO: Convert the last batch of images back to original shape

    # TODO: Convert the last batch of predictions to the original image shape

    # TODO: Plot images

    plt.show()

def intersection_over_union(prediction, ground_truth):
    '''
    Computes the intersection over union (IOU) between prediction and ground truth

    Args:
        prediction : numpy
            N x h x w prediction
        ground_truth : numpy
            N x h x w ground truth

    Returns:
        float : intersection over union
    '''

    # TODO: Computes intersection over union score
    # Implement ONLY if you are working on semantic segmentation

    return 0.0

def peak_signal_to_noise_ratio(prediction, ground_truth):
    '''
    Computes the peak signal to noise ratio (PSNR) between prediction and ground truth

    Args:
        prediction : numpy
            N x h x w prediction
        ground_truth : numpy
            N x h x w ground truth

    Returns:
        float : peak signal to noise ratio
    '''

    # TODO: Computes peak signal to noise ratio
    # Implement ONLY if you are working on image reconstruction or denoising

    return 0.0

def mean_squared_error(prediction, ground_truth):
    '''
    Computes the mean squared error (MSE) between prediction and ground truth

    Args:
        prediction : numpy
            N x h x w prediction
        ground_truth : numpy
            N x h x w ground truth

    Returns:
        float : mean squared error
    '''

    # TODO: Computes mean squared error
    # Implement ONLY if you are working on image reconstruction or denoising

    return 0.0

def plot_images(X, Y, n_row, n_col, fig_title):
    '''
    Creates n_row by n_col panel of images

    Args:
        X : numpy
            N x h x w input data
        Y : numpy
            N x h x w predictions
        n_row : int
            number of rows in figure
        n_col : list[str]
            number of columns in figure
        fig_title : str
            title of plot

        Please add any necessary arguments
    '''

    fig = plt.figure()
    fig.suptitle(fig_title)

    # TODO: Visualize your input images and predictions


if __name__ == '__main__':

    # TODO: Set up data preprocessing step
    # https://pytorch.org/docs/stable/torchvision/transforms.html
    data_preprocess_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])

    # Download and setup your training set
    dataset_train = None

    # Setup a dataloader (iterator) to fetch from the training set
    dataloader_train = None

    # Download and setup your validation/testing set
    dataset_test = None

    # TODO: Setup a dataloader (iterator) to fetch from the validation/testing set
    dataloader_test = None

    # TODO: Define network
    net = None

    # TODO: Setup learning rate optimizer
    # https://pytorch.org/docs/stable/optim.html?#torch.optim.SGD

    if args.train_network:
        # Set network to training mode
        net.train()

        # TODO: Train network and save into checkpoint

        # Saves weight to checkpoint
        torch.save({ 'state_dict' : net.state_dict()}, './checkpoint.pth')
    else:
        # Load network from checkpoint
        checkpoint = torch.load('./checkpoint.pth')
        net.load_state_dict(checkpoint['state_dict'])

    # Set network to evaluation mode
    net.eval()

    # TODO: Evaluate network on testing set
