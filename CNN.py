import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self,
                 image_size: int,
                 num_conv_layers: int=2,
                 kernel_size: int=3,
                 num_classes: int=2):
        super().__init__()
        assert image_size // (2**num_conv_layers) >= kernel_size, \
            "Pooling will reduce size to be smaller than kernel size."
        self.image_size = image_size
        # get convolution layers
        num_in = 3
        num_out = 8
        conv_layers = []
        for _ in range(num_conv_layers):
            conv_layer = nn.Conv2d(in_channels=num_in,
                                   out_channels=num_out,
                                   kernel_size=kernel_size,
                                   padding=kernel_size//2)
            conv_layers.append(conv_layer)
            num_in = num_out
            num_out = num_out * 2
        self.conv_layers = nn.ModuleList(conv_layers)
        # use max pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # ReLU function
        self.relu = nn.ReLU()
        # fully connected layers
        flatten_size = self._get_flattened_size()
        self.fc1 = nn.Linear(flatten_size, 128)
        self.fc2 = nn.Linear(128, num_classes)


    def forward(self, x: torch.Tensor):
        '''
        Arguments:
            x (torch.Tensor): tensor of shape [batch size, channel, width, height]
        
        Returns:
            tensor: tensor of shape [batch size, number of classes]
        '''
        for conv in self.conv_layers:
            x = self.relu(conv(x))
            x = self.pool(x)
        x = x.view(x.size(0), -1)  # shape: [batch size, channel x width x height]
        x = self.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits


    def _get_flattened_size(self): 
        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.image_size, self.image_size)
            for conv in self.conv_layers:
                dummy = self.pool(self.relu(conv(dummy)))
            flatten_size = dummy.view(1, -1).shape[1]
        return flatten_size
