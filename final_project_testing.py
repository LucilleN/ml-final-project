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

IMG_SIZE = 256

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

    def __init__(self, img_dimensions, n_class):
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
        self.img_dimensions = img_dimensions
        self.kernel_size = 3
        self.stride = 2

        """
        Encoding
        """

        # Convolutional 1
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=self.kernel_size, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=self.kernel_size, padding=1)
        
        # Pooling 1
        self.pool1 = torch.nn.MaxPool2d(kernel_size=self.kernel_size, stride=self.stride)  

        # Convolutional 2
        self.conv2_1 = torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=self.kernel_size, padding=1)
        self.conv2_2 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=self.kernel_size, padding=1)
        
        # Pooling 2
        self.pool2 = torch.nn.AvgPool2d(kernel_size=self.kernel_size, stride=self.stride) 

        # Convolutional 3
        self.conv3_1 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=self.kernel_size, padding=1)
        self.conv3_2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=self.kernel_size, padding=1)
        
        # Pooling 3
        self.pool3 = torch.nn.MaxPool2d(kernel_size=self.kernel_size, stride=self.stride) 

        """
        Latent Vector
        """

        # Convolutional 4
        self.conv4_1 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=self.kernel_size, padding=1)
        self.conv4_2 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=self.kernel_size, padding=1)

        """
        Decoding
        """

        # Upsampling / Transpose Convolutional 1
        self.transposed_conv1 = torch.nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=self.kernel_size, stride=self.stride)

        # Convolutional 5
        self.conv5_1 = torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=self.kernel_size, padding=1)
        self.conv5_2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=self.kernel_size, padding=1)
        
        # Upsampling / Transpose Convolutional 2
        self.transposed_conv2 = torch.nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=self.kernel_size, stride=self.stride)

        # Convolutional 6
        self.conv6_1 = torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=self.kernel_size, padding=1)
        self.conv6_2 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=self.kernel_size, padding=1)
        
        # Upsampling / Transpose Convolutional 3
        self.transposed_conv3 = torch.nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=self.kernel_size, stride=self.stride)

        # Convolutional 7
        self.conv7_1 = torch.nn.Conv2d(in_channels=16, out_channels=8, kernel_size=self.kernel_size, padding=1)
        self.conv7_2 = torch.nn.Conv2d(in_channels=8, out_channels=n_class, kernel_size=self.kernel_size, padding=1)

        """
        Fixing the one-pixel mismatch with an upsampling layer
        """
        self.upsample = torch.nn.Upsample((img_dimensions,img_dimensions), mode='bilinear')

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
        h = self.activation_function(self.transposed_conv1(h))
        # Convolutional 5
        h = self.activation_function(self.conv5_1(h))
        h = self.activation_function(self.conv5_2(h))

        # Upsampling / Transpose Convolutional 2
        h = self.activation_function(self.transposed_conv2(h))
        # Convolutional 6
        h = self.activation_function(self.conv6_1(h))
        h = self.activation_function(self.conv6_2(h))

        # Upsampling / Transpose Convolutional 3
        h = self.activation_function(self.transposed_conv3(h))
        # Convolutional 7
        h = self.activation_function(self.conv7_1(h))
        h = self.conv7_2(h)

        # h = self.upsample(h)
        h = torch.nn.functional.interpolate(h, size=(self.img_dimensions, self.img_dimensions), mode='bilinear')

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

            # print("BATCH", batch)
            
            shape = images.shape[2:4]
            images = torch.nn.functional.interpolate(images, (IMG_SIZE, IMG_SIZE))

            # TODO: Forward through the network
            outputs = net(images)

            outputs = torch.nn.functional.interpolate(outputs, (shape[0], shape[1]))

            # print("outputs[0] (a single sample of this batch) is")
            # print(outputs[0].shape)
            # print(outputs[0])
            first_prediction = outputs[0]
            first_groundtruth = labels[0][0]
            # print("unique labels in groundtruth", torch.unique(first_groundtruth))
            first_image = images[0][0]
            
            # outputs = outputs.view(outputs.shape[0], outputs.shape[1], -1)
            # outputs = outputs.detach().numpy()
            # outputs = torch.tensor(np.transpose(outputs, (0, 2, 1)))

            # outputs = outputs.detach().numpy()
            # outputs = torch.tensor(np.transpose(outputs, (0, 2, 3, 1)))
            # print("outputs shape:", outputs.shape)
            # print("outputs unique values are between: ", torch.min(outputs).item(), torch.max(outputs).item())
            # print("outputs[0] (a single sample of this batch) is")
            # print(outputs[0])
            # print("outputs[0][5] is (probabilities for each pixel that it's class 6)")
            # print(outputs[0][5])

            # TODO: Clear gradients so we don't accumlate them from previous batches
            optimizer.zero_grad()

            # TODO: Compute loss function

            # labels = labels.view(labels.shape[0], labels.shape[1], -1)
            labels = torch.squeeze(labels, dim=1)
            labels = torch.round(labels * 255)
            labels[labels==255] = 21
            labels = labels.long()
            # print("labels shape:", labels.shape)
            # print("labels unique values are between: ", torch.min(labels).item(), torch.max(labels).item())
            # print("labels unique values are: ", torch.unique(labels))
            
            loss = loss_func(outputs, labels)

            # if batch > 10:

            #     pred_fig = plt.figure()
            #     for label in range(len(first_prediction)):
            #         score_map = first_prediction[label, ...]
            #         ax = pred_fig.add_subplot(2, 11, label+1)
            #         ax.imshow(score_map.detach().numpy(), vmin=0.0, vmax=1.0)

            #     groundtruth_fig = plt.figure()

            #     ax = groundtruth_fig.add_subplot(1, 1, 1)
            #     ax.imshow(first_groundtruth)

            #     image_fig = plt.figure()
            #     ax = image_fig.add_subplot(1, 1, 1)
            #     ax.imshow(first_image)

            #     plt.show()
            #     break



            # print("LOSS", loss)
            
            # TODO: Update parameters by backpropagation
            loss.backward()
            optimizer.step()

            # TODO: Accumulate total loss for the epoch
            total_loss = total_loss + loss.item()

        mean_loss = total_loss / float(batch)

        # Log average loss over the epoch
        print('Epoch=%d  Loss: %.3f' % (epoch + 1, mean_loss))

    return net

def evaluate(net, dataloader, classes):
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
    n_sample = 0

    cumulative_images = torch.tensor([])
    cumulative_predictions = torch.tensor([])
    cumulative_ground_truths = torch.tensor([])

    ious = 0

    # Make sure we do not backpropagate
    with torch.no_grad():

        for batch, (images, labels) in enumerate(dataloader):
            # print("unique labels:", torch.unique(labels))
            # TODO: Forward through the network
            outputs = net(images)
            # print("eval > outputs shape", outputs.shape)
            # TODO: Take the argmax over the outputs
            _, predictions = torch.max(outputs, dim=1)
            # print("eval > predictions shape", predictions.shape)
            # print("eval > unique predictions")
            # print(torch.unique(predictions))
            # TODO: Reshape labels from (N, , W, C) to (N, H, W)
            ground_truths = torch.squeeze(labels, dim=1)
            ground_truths = torch.round(ground_truths * 255)
            ground_truths[ground_truths==255] = 21

            # Accumulate number of samples
            n_sample = n_sample + labels.shape[0]

            iou = intersection_over_union(predictions, ground_truths)
            ious += iou

            cumulative_images = torch.cat((cumulative_images, images), 0)
            cumulative_predictions = torch.cat((cumulative_predictions, predictions), 0)
            cumulative_ground_truths = torch.cat((cumulative_ground_truths, ground_truths), 0)

    # TODO: Compute mean evaluation metric(s)
    # IOU = intersection_over_union(cumulative_predictions, cumulative_ground_truths)
    # TODO: Print scores
    avg_iou = ious / float(len(dataloader)) 
    print(f'Jaccard Index over {n_sample} images: {avg_iou * 100.0}%')

    # TODO: Convert the last batch of images back to original shape
    cumulative_images = cumulative_images.view(cumulative_images.shape[0], cumulative_images.shape[1], cumulative_images.shape[2], cumulative_images.shape[3])
    cumulative_images = cumulative_images.cpu().numpy()
    cumulative_images = np.transpose(cumulative_images, (0, 2, 3, 1))

    # TODO: Convert the last batch of predictions to the original image shape
    # stacking 3 predictions together along the 1 axis to then convert image back to original image shape
    cumulative_predictions = torch.stack((cumulative_predictions, cumulative_predictions, cumulative_predictions), dim=1) 
    cumulative_predictions = cumulative_predictions.view(cumulative_predictions.shape[0], cumulative_predictions.shape[1], cumulative_predictions.shape[2], cumulative_predictions.shape[3])
    cumulative_predictions = cumulative_predictions.cpu().numpy()
    cumulative_predictions = np.transpose(cumulative_predictions, (0, 2, 3, 1))
    # TODO: Plot images
    plot_images(cumulative_images, cumulative_predictions, n_row=1, n_col=1, fig_title='VOC 2012 Classification Results', classes=classes)  
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

    intersection = torch.sum(torch.where(prediction==ground_truth, 1, 0))
    union = 2 * torch.prod(torch.tensor(prediction.shape)) - intersection
    return float(intersection) / float(union)


    # labels = torch.unique(ground_truth)
    # print("unique labels:")
    # print(labels)
    # print("unique predictions:")
    # print(torch.unique(prediction))
    # ious = 0.0
    # for label in labels:
    #     print("calculating iou for label", label)
    #     intersection = torch.sum((prediction==ground_truth) * (ground_truth==label))
    #     union = torch.sum((prediction==label) + (ground_truth==label))
    #     iou = float(intersection / union)
    #     print("intersection:", intersection, "union:", union, "single iou:", iou)
    #     ious += iou
    # # J(A,B) = |A && B| / |A U B| == |A && B| / (|A|+|B| - |A&&B|)
    # avg_iou = float(ious / len(labels))

    # print('IOU: ', avg_iou)
    # return avg_iou  

def plot_images(X, Y, n_row, n_col, fig_title, classes):
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
    import matplotlib.patches as mpatches

    fig = plt.figure()
    fig.suptitle(fig_title)

    color_dict = {
        '0' : [0/255, 0/255, 0/255],
        '1' : [255/255, 0/255, 0/255],
        '2' : [0/255, 255/255, 0/255],
        '3' : [0/255, 0/255, 255/255],
        '4' : [255/255, 255/255, 0/255],
        '5' : [0/255, 255/255, 255/255],
        '6' : [255/255, 0/255, 255/255],
        '7' : [255/255, 69/255, 90/255],
        '8' : [128/255, 0/255, 128/255],
        '9' : [0/255, 100/255, 0/255],
        '10' : [0/255, 0/255, 128/255],
        '11' : [0/255, 128/255, 128/255],
        '12' : [128/255, 128/255, 128/255],
        '13' : [128/255, 0/255, 0/255],
        '14' : [255/255, 20/255, 147/255],
        '15' : [139/255, 69/255, 19/255],
        '16' : [218/255, 165/255, 32/255],
        '17' : [123/255, 104/255, 238/255],
        '18' : [250/255, 128/255, 114/255],
        '19' : [221/255, 160/255, 221/255],
        '20' : [127/255, 255/255, 212/255],
        '21' : [255/255, 255/255, 255/255],
    }

    # TODO: Visualize your input images and predictions
    for i in range(1, n_row * n_col + 1):

        ax = fig.add_subplot(n_row, n_col, i)
        index = i - 1

        x_i = X[i, ...]
        y_i = Y[i, ...]  
        for i, h in enumerate(y_i):
            for j, w in enumerate(h):
                y_i[i][j] = color_dict[str(int(y_i[i][j][0]))]
        
        visual = np.concatenate((x_i, y_i), axis=1)

        if len(visual.shape) == 1:
            visual = np.expand_dims(visual, axis=0)
            
        ax.imshow(visual)
 
        plt.box(False)
        plt.axis('off')

    # adding legend
    handles = []
    for i in range(len(classes)):
        handles.append(mpatches.Patch(color=color_dict[str(i)], label=classes[i]))

    plt.legend(handles=handles, title='Legend', bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=4)
    plt.tight_layout()


if __name__ == '__main__':


    # TODO: Set up data preprocessing step
    # https://pytorch.org/docs/stable/torchvision/transforms.html
    data_preprocess_transform = torchvision.transforms.Compose([
        # image size should be divisble by 2^(number of max pools)
        # torchvision.transforms.Resize((IMG_SIZE,IMG_SIZE)),
        torchvision.transforms.CenterCrop((IMG_SIZE,IMG_SIZE)),
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

    # Download and setup your validation/testing set
    dataset_test = torchvision.datasets.VOCSegmentation(
        root='./data',
        year= '2012',
        image_set= 'val',
        download=True,
        transform=data_preprocess_transform,
        target_transform=data_preprocess_transform)

    # TODO: Setup a dataloader (iterator) to fetch from the validation/testing set
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2)

    # Define the possible classes in VOC 2012 dataset
    classes = [
        'background',
        'aeroplane',
        'bicycle', 
        'bird', 
        'boat',
        'bottle', 
        'bus', 
        'car',
        'cat',
        'chair', 
        'cow', 
        'dining table', 
        'dog', 
        'horse', 
        'motorbike',
        'person',
        'potted plant', 
        'sheep', 
        'sofa', 
        'train', 
        'tv/monitor',
        'unlabeled'
    ]
    # first_batch_imgs, first_batch_labels = list(dataloader_train)[0]
    # first_img = first_batch_imgs[0]
    # img_shape = first_img.shape
    # print("IMG SHAPE", img_shape)

    # VOC 2012 dataset has 22 classes
    n_class = 22

    # TODO: Define network
    net = FullyConvolutionalNetwork(
        # n_input_feature=num_features,
        img_dimensions=IMG_SIZE,
        n_class=n_class)

    # TODO: Setup learning rate optimizer
    # https://pytorch.org/docs/stable/optim.html?#torch.optim.SGD
    optimizer = torch.optim.SGD(
    # optimizer = torch.optim.Adam(
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
        dataloader=dataloader_test,
        classes=classes)