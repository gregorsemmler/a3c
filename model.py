import numpy as np
import torch
from torch import nn


def get_output_shape(layer, shape):
    layer_training = layer.training
    if layer_training:
        layer.eval()
    out = layer(torch.zeros(1, *shape))
    before_flattening = tuple(out.size())[1:]
    after_flattening = int(np.prod(out.size()))
    if layer_training:
        layer.train()
    return before_flattening, after_flattening


class AtariModel(nn.Module):

    def __init__(self, input_shape, n_actions, conv_details=((16, 8, 4, 0), (32, 4, 2, 0)), fully_details=(256,)):
        super().__init__()
        self.input_shapes = input_shape
        self.n_actions = n_actions

        prev_n_filters = self.input_shapes[0]
        conv_layers = []
        for (n_filters, k_size, stride, padding) in conv_details:
            conv_layers.append(nn.Conv2d(prev_n_filters, n_filters, kernel_size=k_size, stride=stride, padding=padding))
            conv_layers.append(nn.ReLU(inplace=True))
            prev_n_filters = n_filters

        self.conv = nn.Sequential(*conv_layers)

        _, prev_full_n = get_output_shape(self.conv, input_shape)
        policy_full_layers = [nn.Flatten()]
        value_full_layers = [nn.Flatten()]
        for full_n in fully_details:
            policy_full_layers.append(nn.Linear(prev_full_n, full_n))
            policy_full_layers.append(nn.ReLU(inplace=True))
            value_full_layers.append(nn.Linear(prev_full_n, full_n))
            value_full_layers.append(nn.ReLU(inplace=True))
            prev_full_n = full_n

        policy_full_layers.append(nn.Linear(prev_full_n, n_actions))
        value_full_layers.append(nn.Linear(prev_full_n, 1))

        self.policy_head = nn.Sequential(*policy_full_layers)
        self.value_head = nn.Sequential(*value_full_layers)

    def forward(self, x):
        conv_out = self.conv(x)
        return self.policy_head(conv_out), self.value_head(conv_out)


class ResidualBlock(nn.Module):

    def __init__(self, in_planes, out_planes, bias=True, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, padding=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.down_sample = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias),
            nn.BatchNorm2d(out_planes)
        ) if stride != 1 or in_planes != out_planes else lambda x: x
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.down_sample(x)

        out += identity
        out = self.relu(out)

        return out


class ResidualModel(nn.Module):

    def __init__(self, input_shape, num_filters, num_residual_blocks, val_hidden_size, num_actions):
        super().__init__()
        self.input_shape = input_shape
        self.num_filters = num_filters
        self.num_residual_blocks = num_residual_blocks
        self.val_hidden_size = val_hidden_size
        self.num_actions = num_actions
        self.residual_tower = nn.Sequential(
            nn.Conv2d(self.input_shape[0], self.num_filters, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.num_filters),
            nn.ReLU(inplace=True),
            *[ResidualBlock(self.num_filters, self.num_filters) for _ in range(num_residual_blocks)]
        )

        tower_out_shape = (self.num_filters,) + self.input_shape[1:]

        self.policy_conv = nn.Sequential(
            nn.Conv2d(self.num_filters, 2, kernel_size=1, stride=1),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True)
        )
        _, poly_conv_flat = get_output_shape(self.policy_conv, tower_out_shape)
        self.policy_head = nn.Sequential(
            self.policy_conv,
            nn.Flatten(),
            nn.Linear(poly_conv_flat, self.num_actions)
        )

        self.val_conv = nn.Sequential(
            nn.Conv2d(self.num_filters, 1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )
        _, val_conv_flat = get_output_shape(self.val_conv, tower_out_shape)
        self.val_head = nn.Sequential(
            self.val_conv,
            nn.Flatten(),
            nn.Linear(val_conv_flat, self.val_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.val_hidden_size, 1),
            nn.Tanh()
        )

    def forward(self, x):
        tower_out = self.residual_tower(x)
        return self.policy_head(tower_out), self.val_head(tower_out)
