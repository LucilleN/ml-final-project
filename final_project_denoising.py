'''
Names (Please write names in <Last Name, First Name> format):
1. Njoo, Lucille
2. Kane, Louis
3. Arteaga, Andrew 

Project type:
    Image Denoising with VOC 2012 Dataset

Report what each member did in this project:
    Lucille: Network Architecture, Training, and Evaluation
    Andrew: Image visualizations, Data Plotting, and Evaluation 
    Louis: Network Architecture and Tuning

'''
import argparse
import math
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


args = parser.parse_args()


class FullyConvolutionalNetwork(nn.Module):
    '''
    Fully convolutional network

    Args:
        img_width: int
            Width of the images to denoise, in pixels
        img_height: int
            Height of the images to denoise, in pixels
        sigma: float
            Amount of noise to add to images. Use sigma=0.0 for only image reconstruction.
    '''

    def __init__(self, img_width, img_height, sigma):
        super(FullyConvolutionalNetwork, self).__init__()
        self.activation_function = torch.nn.Tanh()
        self.img_width = img_width
        self.img_height = img_height
        self.kernel_size = 3
        self.stride = 2
        self.sigma = sigma

        # Convolutional 1
        self.conv1_2 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=self.kernel_size, padding=1)
        
        # Pooling 1
        self.pool1 = torch.nn.MaxPool2d(kernel_size=self.kernel_size, stride=self.stride)  

        # Convolutional 2
        self.conv2_2 = torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=self.kernel_size, padding=1)
        
        # Upsampling / Transpose 
        self.transposed_conv3 = torch.nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=self.kernel_size, stride=self.stride)

        # Convolutional 3
        self.conv3_2 = torch.nn.Conv2d(in_channels=24, out_channels=3, kernel_size=self.kernel_size, padding=1)


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
        
        # Convolutional 1
        # h = self.activation_function(self.conv1_1(h))
        h = self.activation_function(self.conv1_2(h))
        first_output = torch.nn.functional.interpolate(h, size=(255, 255), mode='bilinear')
        
        # Pooling 1
        h = self.pool1(h)

        # Convolutional 2
        # h = self.activation_function(self.conv2_1(h))
        h = self.activation_function(self.conv2_2(h))

        # Upsampling / Transpose
        h = self.activation_function(self.transposed_conv3(h))
        # transpose_output = torch.nn.functional.interpolate(h, size=(self.img_height, self.img_width), mode='bilinear')
        transpose_output = h
        new_output = torch.cat((first_output, transpose_output), dim=1)
        
        # Convolutional 3
        # h = self.activation_function(self.conv3_1(h))
        h = nn.functional.sigmoid(self.conv3_2(new_output))
        # h = nn.functional.sigmoid(self.conv3_2(h))

        # h = self.upsample(h)
        h = torch.nn.functional.interpolate(h, size=(self.img_height, self.img_width), mode='bilinear')

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

    Returns:
        torch.nn.Module : trained network
    '''

    # TODO: Define cross entropy loss
    # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    loss_func = torch.nn.MSELoss()
    
    for epoch in range(n_epoch):

        # Accumulate total loss for each epoch
        total_loss = 0.0

        # TODO: Decrease learning rate when learning rate decay period is met
        # e.g. decrease learning rate by a factor of decay rate every 2 epoch
        if epoch and epoch % learning_rate_decay_period == 0:

            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate_decay * param_group['lr']

        for batch, (clean_images, _) in enumerate(dataloader):

            # Replace randn below with just rand in order to use uniform distribution for noise
            noise = net.sigma * torch.randn(clean_images.shape)
            noisy_images = torch.clamp(clean_images + noise, min=0, max=1)

            # TODO: Forward through the network
            outputs = net(noisy_images)

            # TODO: Clear gradients so we don't accumlate them from previous batches
            optimizer.zero_grad()

            # TODO: Compute loss function
            loss = loss_func(outputs, clean_images)
            
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
    '''
    n_sample = 0

    cumulative_noisy_images = torch.tensor([])
    cumulative_denoised_images = torch.tensor([])
    cumulative_clean_images = torch.tensor([])

    mses = 0.0
    psnrs = 0.0 

    # Make sure we do not backpropagate
    with torch.no_grad():

        for batch, (clean_images, _) in enumerate(dataloader):

            # Replace randn below with just rand in order to use uniform distribution for noise
            noise = net.sigma * torch.randn(clean_images.shape)
            noisy_images = torch.clamp(clean_images + noise, min=0, max=1)

            # TODO: Forward through the network
            outputs = net(noisy_images)

            # Accumulate number of samples
            n_sample = n_sample + clean_images.shape[0]

            # Accumulate MSE and PSNR metrics for the current batch
            mse = mean_squared_error(outputs, clean_images)
            mses += mse
            psnr = peak_signal_to_noise_ratio(outputs, clean_images)
            psnrs += psnr

            cumulative_noisy_images = torch.cat((cumulative_noisy_images, noisy_images), 0)
            cumulative_denoised_images = torch.cat((cumulative_denoised_images, outputs), 0)

    # TODO: Compute mean evaluation metric(s)
    avg_mse = mses / float(len(dataloader)) 
    avg_psnr = psnrs / float(len(dataloader)) 

    # TODO: Print scores
    print(f'MSE score over {n_sample} images: {avg_mse}')
    print(f'PSNR over {n_sample} images: {avg_psnr}')

    # Reshape images from (N x C x H x W) to (N x H x W x C) for imshow
    cumulative_noisy_images = cumulative_noisy_images.cpu().numpy()
    cumulative_noisy_images = np.transpose(cumulative_noisy_images, (0, 2, 3, 1))
    cumulative_denoised_images = cumulative_denoised_images.cpu().numpy()
    cumulative_denoised_images = np.transpose(cumulative_denoised_images, (0, 2, 3, 1))
    
    # TODO: Plot images
    plot_images(cumulative_noisy_images, cumulative_denoised_images, n_row=2, n_col=1, fig_title='VOC 2012 Image Denoising Results')  
    plt.show()


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

    # Computes peak signal to noise ratio
    
    MAX_INTENSITY = 255
    prediction = prediction * MAX_INTENSITY
    ground_truth = ground_truth * MAX_INTENSITY

    MSE = mean_squared_error(prediction, ground_truth)

    # PSNR = 20 * log_10 (MAX_INTENSITY) - 10 * log_10 (MSE)
    PSNR = 20 * math.log10(MAX_INTENSITY) - 10 * math.log10(MSE)

    return PSNR


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
    MSE = torch.mean((prediction - ground_truth) ** 2)

    return MSE


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
        n_col : int
            number of columns in figure
        fig_title : str
            title of plot

        Please add any necessary arguments
    '''
    import matplotlib.patches as mpatches

    fig = plt.figure()
    fig.suptitle(fig_title)

    # TODO: Visualize your input images and predictions
    for i in range(1, n_row * n_col + 1):

        ax = fig.add_subplot(n_row, n_col, i)
        index = i - 1

        x_i = X[i, ...]
        y_i = Y[i, ...]  
        
        visual = np.concatenate((x_i, y_i), axis=1)

        if len(visual.shape) == 1:
            visual = np.expand_dims(visual, axis=0)
            
        ax.imshow(visual)
 
        plt.box(False)
        plt.axis('off')


if __name__ == '__main__':

    IMG_HEIGHT = 256
    IMG_WIDTH = 256

    # TODO: Set up data preprocessing step
    # https://pytorch.org/docs/stable/torchvision/transforms.html
    data_preprocess_transform_train = torchvision.transforms.Compose([
        # image size should be divisble by 2^(number of max pools)
        torchvision.transforms.CenterCrop((IMG_HEIGHT,IMG_WIDTH)),
        torchvision.transforms.ToTensor()
    ])
    data_preprocess_transform_test = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop((IMG_HEIGHT,IMG_WIDTH)),
        torchvision.transforms.ToTensor()
    ])

    # Download and setup your training set
    dataset_train = torchvision.datasets.VOCSegmentation(
        root='./data',
        year= '2012',
        image_set= 'train',
        download=True,
        transform=data_preprocess_transform_train,
        target_transform=data_preprocess_transform_train)

    # Setup a dataloader (iterator) to fetch from the training set
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2)

    # Download and setup your validation/testing set
    dataset_test = torchvision.datasets.VOCSegmentation(
        root='./data',
        year= '2012',
        image_set= 'val',
        download=True,
        transform=data_preprocess_transform_test,
        target_transform=data_preprocess_transform_test)

    # TODO: Setup a dataloader (iterator) to fetch from the validation/testing set
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=2)

    # TODO: Define network
    net = FullyConvolutionalNetwork(
        img_width=IMG_HEIGHT,
        img_height=IMG_WIDTH,
        sigma=0.3)

    # TODO: Setup learning rate optimizer
    # https://pytorch.org/docs/stable/optim.html?#torch.optim.SGD
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.lambda_weight_decay)

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

    # TODO: Evaluate network on testing set
    evaluate(
        net=net,
        dataloader=dataloader_test)