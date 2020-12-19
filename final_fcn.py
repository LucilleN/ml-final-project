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

        # (1) convolutional layers
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        # Convultional + ReLu
        # status: done

        # (2) max pool layers
        # https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.max_pool2d
        # status: done

        # (3) average pool layers
        # https://www.aiworkbox.com/lessons/avgpool2d-how-to-incorporate-average-pooling-into-a-pytorch-neural-network
        # https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.avg_pool2d
        # status: TODO

        # (4) transposed convolutional layers
        # https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
        # status: done

        # https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/models/fcn8s.py
        # Conved args:
        # in_channels: # of channels in the input image
        # out_channels: # of channels produced by the convolution
        # kernal_size: size of the convolving kernel
        # stride: Stride of the convolution. default 1
        # dilation: spacing between kernel elements: default 1
        # padding: zeriod - padding added to both sides of the input: default 0

        # conv1
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=100)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32

        # TODO: avgpool2d layers
        # avgP6
        self.conv6_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu6_1 = nn.ReLU(inplace=True)
        self.conv6_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu6_2 = nn.ReLU(inplace=True)
        self.conv6_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu6_3 = nn.ReLU(inplace=True)
        self.pool6 = nn.AvgPool2d(2, stride=2, ceil_mode=True)  # 1/32

        # avgP7
        self.conv7_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu7_1 = nn.ReLU(inplace=True)
        self.conv7_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu7_2 = nn.ReLU(inplace=True)
        self.conv7_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu7_3 = nn.ReLU(inplace=True)
        self.pool7 = nn.AvgPool2d(2, stride=2, ceil_mode=True)  # 1/32

        # fully connect layers
        # fc8
        self.fc8 = nn.Conv2d(512, 4096, 7)
        self.relu8 = nn.ReLU(inplace=True)
        self.drop8 = nn.Dropout2d()

        # fc9
        self.fc9 = nn.Conv2d(4096, 4096, 1)
        self.relu9 = nn.ReLU(inplace=True)
        self.drop9 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(4096, n_class, 1)
        self.score_pool3 = nn.Conv2d(256, n_class, 1)
        self.score_pool4 = nn.Conv2d(512, n_class, 1)

        self.upscore2 = nn.ConvTranspose2d(
            n_class, n_class, 4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(
            n_class, n_class, 16, stride=8, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(
            n_class, n_class, 4, stride=2, bias=False)

     
    def forward(self, x):
        # https://pytorch.org/docs/stable/nn.functional.html 
        # https://deeplizard.com/learn/video/MasG7tZj-hw
        # conv2d
        '''
            Args:
                x : torch.Tensor
                    tensor of N x d

            Returns:
                torch.Tensor
                    tensor of n_output
        '''

        # TODO: Implement forward function

        h = x
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)
        pool3 = h  # 1/8

        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)
        pool4 = h  # 1/16

        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)

        # average pool
        h = self.relu6_1(self.conv6_1(h))
        h = self.relu6_2(self.conv6_2(h))
        h = self.relu6_3(self.conv6_3(h))
        h = self.pool6(h)

        h = self.relu7_1(self.conv7_1(h))
        h = self.relu7_2(self.conv7_2(h))
        h = self.relu7_3(self.conv7_3(h))
        h = self.pool7(h)

        # fully connected 
        h = self.relu8(self.fc8(h))
        h = self.drop8(h)

        h = self.relu9(self.fc9(h))
        h = self.drop9(h)

        h = self.score_fr(h)
        h = self.upscore2(h)
        upscore2 = h  # 1/16

        h = self.score_pool4(pool4)
        h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
        score_pool4c = h  # 1/16

        h = upscore2 + score_pool4c  # 1/16
        h = self.upscore_pool4(h)
        upscore_pool4 = h  # 1/8

        h = self.score_pool3(pool3)
        h = h[:, :,
              9:9 + upscore_pool4.size()[2],
              9:9 + upscore_pool4.size()[3]]
        score_pool3c = h  # 1/8

        h = upscore_pool4 + score_pool3c  # 1/8

        h = self.upscore8(h)
        h = h[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]].contiguous()

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

        # TODO: REPLACE (IMAGE, LABELS) with appropriate iterators
        for batch, (images, labels) in enumerate(dataloader):

            # TODO: Vectorize images from (N, H, W, C) to (N, d)
            n_dim = np.prod(images.shape[1:])
            images = images.view(-1, n_dim)

            # TODO: Forward through the network
            outputs = net(images)

            # TODO: Clear gradients so we don't accumlate them from previous batches
            optimizer.zero_grad()

            # TODO: Compute loss function
            loss = loss_func(outputs, labels)

            # TODO: Update parameters by backpropagation
            loss.backward()
            optimizer.step()

            # TODO: Accumulate total loss for the epoch
            total_loss = total_loss + loss.item()

        mean_loss = total_loss / float(batch)

        # Log average loss over the epoch
        print('Epoch=%d  Loss: %.3f' % (epoch + 1, mean_loss))

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
    correct = 0
    sample = 0
    # Make sure we do not backpropagate
    with torch.no_grad():

        for (images, labels) in dataloader:

            # TODO: Forward through the network

            # TODO: Compute evaluation metric(s) for each sample

            # TODO: Vectorize images from (N, H, W, C) to (N, d)
            shape = images.shape
            n_dim = np.prod(shape[1:])
            images = images.view(-1, n_dim)

            # TODO: Forward through the network
            outputs = net(images)

            # TODO: Take the argmax over the outputs
            _, predictions = torch.max(outputs, dim=1)

            # Accumulate number of samples
            sample = sample + labels.shape[0]

            # TODO: Check if our prediction is correct
            correct = correct + torch.sum(predictions == labels).item()

    # TODO: Compute mean evaluation metric(s)
    mean_accuracy = 100.0 * correct/sample
    # TODO: Print scores
    print('Mean accuracy over %d images: %d %%' % (sample, mean_accuracy))
    # TODO: Convert the last batch of images back to original shape
    images = images.view(shape[0], shape[1], shape[2], shape[3])
    images = images.cpu().numpy()
    images = np.transpose(images, (0, 2, 3, 1))
    # TODO: Convert the last batch of predictions to the original image shape

    # TODO: Plot images

    plt.show()

def intersection_over_union(prediction, ground_truth):
    # https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    # https://github.com/amdegroot/ssd.pytorch/blob/master/layers/box_utils.py#L48
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
    # J(A,B) = |A && B| / |A U B| == |A && B| / (|A|+|B| - |A&&B|)

    # TODO: Computes intersection over union score
    # Implement ONLY if you are working on semantic segmentation
    A = prediction.size(0)
    B = ground_truth.size(0)

    max_xy = torch.min(prediction[:, 2:].unsqueeze(1).expand(A, B, 2),
                       ground_truth[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(prediction[:, :2].unsqueeze(1).expand(A, B, 2),
                       ground_truth[:, :2].unsqueeze(0).expand(A, B, 2))

    inter = torch.clamp((max_xy - min_xy), min=0)

    inter = inter[:, :, 0] * inter[:, :, 1]

    area_a = ((prediction[:, 2]-prediction[:, 0]) *
              (prediction[:, 3]-prediction[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((ground_truth[:, 2]-ground_truth[:, 0]) *
              (ground_truth[:, 3]-ground_truth[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


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

    for i in range(1, n_row * n_col + 1):

        ax = fig.add_subplot(n_row, n_col, i)

        index = i - 1
        x_i = X[index, ...]
        subplot_title_i = subplot_titles[index]

        if len(x_i.shape) == 1:
            x_i = np.expand_dims(x_i, axis=0)

        ax.set_title(subplot_title_i)
        ax.imshow(x_i)

        plt.box(False)
        plt.axis('off')

    # TODO: Visualize your input images and predictions


if __name__ == '__main__':

    # TODO: Set up data preprocessing step
    # https://pytorch.org/docs/stable/torchvision/transforms.html
    data_preprocess_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])

    # Download and setup your training set

    dataset_train = torchvision.datasets.VOCSegmentation(
        root='./data',
        year= '2012',
        image_set= 'train',
        download=True,
        transform=data_preprocess_transform)

    # dataset_train = torchvision.datasets.Cityscapes(
    #     root='./data',
    #     train=True,
    #     download=True,
    #     transform=data_preprocess_transform)

    # Setup a dataloader (iterator) to fetch from the training set
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2)

    # Download and setup your validation/testing set

    dataset_test =  torchvision.datasets.VOCSegmentation(
        root='./data',
        year= '2012',
        image_set= 'val',
        download=True,
        transform=data_preprocess_transform)

    # dataset_test =  torchvision.datasets.Cityscapes(
    #     root='./data',
    #     train=False,
    #     download=True,
    #     transform=data_preprocess_transform)

    # TODO: Setup a dataloader (iterator) to fetch from the validation/testing set
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2)

    # TODO: Define network

    net = NeuralNetwork(
        n_input_feature=n_input_feature,
        n_output=n_class)

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
        # torch.save({ 'state_dict' : net.state_dict()}, './checkpoint.pth')

        # Saves weight to checkpoint
        torch.save({ 'state_dict' : net.state_dict()}, './checkpoint.pth')
    else:
        # Load network from checkpoint
        checkpoint = torch.load('./checkpoint.pth')
        net.load_state_dict(checkpoint['state_dict'])

    # Set network to evaluation mode
    net.eval()

    # TODO: Evaluate network on testing set

    # evaluate(
    #     net=net,
    #     dataloader=dataloader_test,
    #     classes=classes)