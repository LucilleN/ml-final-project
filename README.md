# ml-final-project
Building fully convolutional neural networks in PyTorch.

# Todo
1. data loading
2. Preprocessing
3. define model
4. construct loss func
5. select optimization algo
6. set up eval metrics
7. training / testing methods

# Source Data
1. https://pytorch.org/docs/stable/torchvision/datasets.html#cityscapes
2. https://www.cityscapes-dataset.com/downloads/
3. https://github.com/mcordts/cityscapesScripts

# Resources
1. https://github.com/SatyamGaba/semantic_segmentation_cityscape
2. https://github.com/nyoki-mtl/pytorch-segmentation
3. https://github.com/meetshah1995/pytorch-semseg
4. https://paperswithcode.com/sota/ semantic-segmentation-on-cityscapes
5. https://discuss.pytorch.org/t/
    best-available-semantic-segmentation-in-pytorch/13107
6. https://meetshah1995.github.io/semantic-segmentation/deep-learning/pytorch/visdom/2017/06/01/semantic-segmentation-over-the-years.html
7. https://www.learnopencv.com/pytorch-for-beginners-semantic-segmentation-using-torchvision/
8. https://github.com/hoya012/semantic-segmentation-tutorial-pytorch
9. https://medium.com/pytorch/accelerate-your-hyperparameter-optimization-with-pytorchs-ecosystem-tools-bc17001b9a49
10. https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
11. https://medium.com/analytics-vidhya/creating-a-very-simple-u-net-model-with-pytorch-for-semantic-segmentation-of-satellite-images-223aa216e705
12. https://github.com/pochih/FCN-pytorch/blob/master/python/fcn.py
13. https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/models/fcn8s.py
14. https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
15. https://github.com/amdegroot/ssd.pytorch/blob/master/layers/box_utils.py#L48

# Notes
1. CONV2D
    a. https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    b. https://www.programcreek.com/python/example/107691/torch.nn.Conv2d
2. Max_Pool
    a. https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    b.
3. Average Pool
    a. https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.avg_pool2d
    b.
4. ConvTranspoose2D
    a. https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
    b.
5. Optimizer
    a. https://pytorch.org/docs/stable/optim.html
    b.
6. Transforms
    a. https://pytorch.org/docs/stable/torchvision/transforms.html
    b.
7. Torch sgd
    a. https://pytorch.org/docs/stable/optim.html?#torch.optim.SGD
    b. 