'''
Names (Please write names in <Last Name, First Name> format):
1. Njoo, Lucille
2. Kane, Louis
3. Arteaga, Andrew 

TODO: Project type
Semantic Segmentation with VOC 2012 Dataset

TODO: Report what each member did in this project

'''
import argparse
import torch, torchvision
import torch.nn as nn
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


class FullyConvolutionalNetwork(nn.Module):
    '''
    Fully convolutional network

    Args:
        Please add any necessary arguments
    '''

    def __init__(self, n_class):
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
        
        self.activation_function = torch.nn.ReLU(inplace=True)
        self.kernel_size = 3

        """
        Encoding
        """

        # Convolutional 1
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=self.kernel_size, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=self.kernel_size, padding=1)
        
        # Pooling 1
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)  

        # Convolutional 2
        self.conv2_1 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=self.kernel_size, padding=1)
        self.conv2_2 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=self.kernel_size, padding=1)
        
        # Pooling 2
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2) 

        # Convolutional 3
        self.conv3_1 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=self.kernel_size, padding=1)
        self.conv3_2 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=self.kernel_size, padding=1)
        
        # Pooling 3
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2) 

        """
        Latent Vector
        """

        # Convolutional 4
        self.conv4_1 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=self.kernel_size, padding=1)
        self.conv4_2 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=self.kernel_size, padding=1)

        """
        Decoding
        """

        # Upsampling / Transpose Convolutional 1
        self.transposed_conv1 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=self.kernel_size, output_padding=1)

        # Convolutional 5
        self.conv5_1 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=self.kernel_size, padding=1)
        self.conv5_2 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=self.kernel_size, padding=1)
        
        # Upsampling / Transpose Convolutional 2
        self.transposed_conv2 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=self.kernel_size, output_padding=1)

        # Convolutional 6
        self.conv6_1 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=self.kernel_size, padding=1)
        self.conv6_2 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=self.kernel_size, padding=1)
        
        # Upsampling / Transpose Convolutional 3
        self.transposed_conv3 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=self.kernel_size, output_padding=1)

        # Convolutional 7
        self.conv7_1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=self.kernel_size, padding=1)
        self.conv7_2 = nn.Conv2d(in_channels=64, out_channels=n_class, kernel_size=self.kernel_size, padding=1)


    def forward(self, x):
        '''
            Args:
                x : torch.Tensor
                    tensor of N x d

            Returns:
                torch.Tensor
                    tensor of n_output
        '''

        h = x
        
        """
        Encoding
        """

        # Convolutional 1
        h = self.activation_function(self.conv1_1(h))
        h = self.activation_function(self.conv1_2(h))
        # Pooling 1
        h = self.pool1(h)

        # Convolutional 2
        h = self.activation_function(self.conv2_1(h))
        h = self.activation_function(self.conv2_2(h))
        # Pooling 2
        h = self.pool2(h)

        # Convolutional 3
        h = self.activation_function(self.conv3_1(h))
        h = self.activation_function(self.conv3_2(h))
        # Pooling 3
        h = self.pool3(h)

        """
        Latent Vector
        """

        # Convolutional 4
        h = self.activation_function(self.conv4_1(h))
        h = self.activation_function(self.conv4_2(h))

        """
        Decoding
        """

        # Upsampling / Transpose Convolutional 1
        h = self.transposed_conv1(h)
        # Convolutional 5
        h = self.activation_function(self.conv5_1(h))
        h = self.activation_function(self.conv5_2(h))

        # Upsampling / Transpose Convolutional 2
        h = self.transposed_conv2(h)
        # Convolutional 6
        h = self.activation_function(self.conv6_1(h))
        h = self.activation_function(self.conv6_2(h))

        # Upsampling / Transpose Convolutional 3
        h = self.transposed_conv3(h)
        # Convolutional 7
        h = self.activation_function(self.conv7_1(h))
        h = self.activation_function(self.conv7_2(h))

        return h


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

    # TODO: Define cross entropy loss
    # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    loss_func = torch.nn.CrossEntropyLoss()

    for epoch in range(n_epoch):

        # Accumulate total loss for each epoch
        total_loss = 0.0

        # TODO: Decrease learning rate when learning rate decay period is met
        # e.g. decrease learning rate by a factor of decay rate every 2 epoch
        if epoch and epoch % learning_rate_decay_period == 0:

            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate_decay * param_group['lr']

        for batch, (images, labels) in enumerate(dataloader):

            # TODO: Vectorize images from (N, H, W, C) to (N, d)
            print("images.shape originally is", images.shape)
            # n_dim = np.prod(images.shape[1:])
            # images = images.view(-1, n_dim)
            # print("images.shape changed to", images.shape)
            print("labels.shape originally is", labels.shape)
            # n_label_dim = np.prod(labels.shape[1:])
            # labels = labels.view(-1, n_label_dim)
            # print("labels.shape changed to", labels.shape)

            # TODO: Forward through the network
            outputs = net(images)

            # TODO: Clear gradients so we don't accumlate them from previous batches
            optimizer.zero_grad()

            # TODO: Compute loss function
            print("about to compute loss")
            # outputs = torch.flatten(outputs)
            # labels = torch.flatten(labels)
            print("output shape:", outputs.shape)
            print("labels shape:", labels.shape)
            loss = loss_func(outputs, labels)

            # TODO: Update parameters by backpropagation
            loss.backward()
            optimizer.step()

            # TODO: Accumulate total loss for the epoch
            total_loss = total_loss + loss.item()

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
    n_correct = 0
    n_sample = 0
    # Make sure we do not backpropagate
    with torch.no_grad():

        for (images, labels) in dataloader:

            # TODO: Vectorize images from (N, H, W, C) to (N, d)
            shape = images.shape
            n_dim = np.prod(shape[1:])
            images = images.view(-1, n_dim)

            # TODO: Forward through the network
            outputs = net(images)

            # TODO: Take the argmax over the outputs
            _, predictions = torch.max(outputs, dim=1)

            # Accumulate number of samples
            n_sample = n_sample + labels.shape[0]

            # TODO: Check if our prediction is correct
            n_correct = n_correct + torch.sum(predictions == labels).item()

    # TODO: Compute mean evaluation metric(s)
    mean_accuracy = 100.0 * n_correct / n_sample
    # IOU = intersection_over_union(predictions, images)
    # TODO: Print scores
    # print('Jaccard Index over %d images: %d %%' % (n_sample, IOU))
    print('Mean accuracy over %d images: %d %%' % (n_sample, mean_accuracy))
    # TODO: Convert the last batch of images back to original shape
    images = images.view(shape[0], shape[1], shape[2], shape[3])
    images = images.cpu().numpy()
    images = np.transpose(images, (0, 2, 3, 1))
    # TODO: Convert the last batch of predictions to the original image shape
    predictions = predictions.view(shape[0], shape[1], shape[2], shape[3])
    predictions = predictions.cpu().numpy()
    predictions = np.transpose(predictions, (0, 2, 3, 1))
    # TODO: Plot images
    # plot_images(images, predictions, n_row=2, n_col= int(images.shape[0] / 2), fig_title='VOC 2012 Classification Results')  
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
        torchvision.transforms.Resize((32,32)),
        torchvision.transforms.ToTensor()
    ])

    # Download and setup your training set
    dataset_train = torchvision.datasets.VOCSegmentation(
        root='./data',
        year= '2012',
        image_set= 'train',
        download=True,
        transform=data_preprocess_transform,
        target_transform=data_preprocess_transform)

    # Setup a dataloader (iterator) to fetch from the training set
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2)

    # # Download and setup your validation/testing set
    # dataset_test = torchvision.datasets.VOCSegmentation(
    #     root='./data',
    #     year= '2012',
    #     image_set= 'val',
    #     download=True,
    #     transform=data_preprocess_transform)

    # # TODO: Setup a dataloader (iterator) to fetch from the validation/testing set
    # dataloader_test = torch.utils.data.DataLoader(
    #     dataset_test,
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     num_workers=2)

    # Define the possible classes in VOC 2012 dataset
    classes = [
        'person',
        'bird', 
        'cat', 
        'cow', 
        'dog', 
        'horse', 
        'sheep', 
        'aeroplane', 
        'bicycle', 
        'boat', 
        'bus', 
        'car',
        'motorbike',
        'train', 
        'bottle', 
        'chair', 
        'dining table', 
        'potted plant', 
        'sofa', 
        'tv/monitor'
    ]

    # Number of input features: 3 (channel) by 32 (height) by 32 (width)
    num_pixels = 32 * 32
    num_features = num_pixels * 3

    # VOC 2012 dataset has 20 classes
    n_class = 20

    # TODO: Define network
    net = FullyConvolutionalNetwork(
        # n_input_feature=num_features,
        n_class=n_class)

    # TODO: Setup learning rate optimizer
    # https://pytorch.org/docs/stable/optim.html?#torch.optim.SGD
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=args.learning_rate,
        weight_decay=args.lambda_weight_decay,
        momentum=args.momentum)

    if args.train_network:
        # Set network to training mode
        net.train()

        # TODO: Train network and save into checkpoint
        net = train(
            net=net,
            dataloader=dataloader_train,
            n_epoch=args.n_epoch,
            optimizer=optimizer,
            learning_rate_decay=args.learning_rate_decay,
            learning_rate_decay_period=args.learning_rate_decay_period)
        # Saves weight to checkpoint
        torch.save({ 'state_dict' : net.state_dict()}, './checkpoint.pth')
    else:
        # Load network from checkpoint
        checkpoint = torch.load('./checkpoint.pth')
        net.load_state_dict(checkpoint['state_dict'])

    # Set network to evaluation mode
    net.eval()

    # # TODO: Evaluate network on testing set
    # evaluate(
    #     net=net,
    #     dataloader=dataloader_test,
    #     classes=classes)