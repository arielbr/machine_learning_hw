""" Model classes defined here! """

import torch
import torch.nn.functional as F


class FeedForward(torch.nn.Module):
    def __init__(self, hidden_dim):
        """
        In the constructor we instantiate two nn.Linear modules and
        assign them as member variables.
        """
        super(FeedForward, self).__init__()
        self.linear1 = torch.nn.Linear(28*28, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, 8)

    def forward(self, x):
        """
        Compute the forward pass of our model, which outputs logits.
        """
        # type(x) = <class 'torch.Tensor'>, torch.Size=([40, 784])
        # TODO: Implement this!
        output1 = self.linear1(x)  # to 100 features of level 1
        output1_activated = F.relu(output1)
        output2 = self.linear2(output1_activated)
        return output2


class SimpleConvNN(torch.nn.Module):
    def __init__(self, n1_chan, n1_kern, n2_kern):
        super(SimpleConvNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, n1_chan, kernel_size=n1_kern)
        self.conv2 = torch.nn.Conv2d(n1_chan, 8, kernel_size=n2_kern, stride=2)

    def forward(self, x):
        # TODO: Implement this!
        # x: 40 * 784, class torch.Tensor
        # TODO n1_chan, n1_kern, n2_kern all = 10, where do they come from? also if image 28*28, how to convert to 10*10
        total_dim = list(x.shape)[1]
        dim_2d = int(total_dim**0.5)
        # convert into an 2d image
        # 40 x 1 x 28 x 28
        # x.view (x.shape[0],1,28,28)
        xx = torch.reshape(x, (list(x.shape)[0], 1, dim_2d, dim_2d))
        # conv1
        output1 = self.conv1(xx)  # TODO expecting 40 x 1 x 10 x 10?
        activated_1 = F.relu(output1)
        # conv2
        output2 = self.conv2(activated_1)
        activated_2 = F.relu(output2)
        # get logits
        # TODO what are the parameters
        # TODO parameters # param (x, x.shape[2]) no stride
        max2dpool = torch.nn.MaxPool2d(7, stride=2)
        # x= torch.nn.MaxPool2d(x, x.shape[2])
        # shape[0], [1]
        output = max2dpool(activated_2)
        output = torch.reshape(output, (40, 8))
        print(output.shape)
        return output


class BestNN(torch.nn.Module):
    # TODO: You can change the parameters to the init method if you need to
    # take hyperparameters from the command line args!
    def __init__(self):
        super(BestNN, self).__init__(optimizer_l1=True,
                                     batch_size=5, layers=2, units_per_layer=10)
        self.conv1 = torch.nn.Conv2d(1, n1_chan, kernel_size=n1_kern)
        self.conv2 = torch.nn.Conv2d(n1_chan, 8, kernel_size=n2_kern, stride=2)
        self.linear1 = torch.nn.Linear(28*28, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, 8)        

        # TODO: Implement this!
        self.optimizer_l1 = optimizer_l1
        self.batch_size = batch_size
        self.layers = layers
        self.units_per_layer = units_per_layer
        #raise NotImplementedError()

    def forward(self, x):
        # TODO: Implement this!
        #raise NotImplementedError()
        output1 = self.conv1(xx)

        # check order, check other parameters see torch.nn other methods

        # accuracy 85
