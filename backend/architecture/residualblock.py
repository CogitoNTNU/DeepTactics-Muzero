import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, channels :int, kernel_size: int, stride : int, padding : int) -> None:
        super(ResBlock, self).__init__()
        self.c1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU()
        self.c2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x : torch.Tensor):
        out= self.c1(x)
        out= self.relu(out)
        out= self.c2(out)
        out= out + x
        return self.relu(out)


if __name__ == "__main__":
    resBlock = ResBlock(6,3,1,1)
    ones = torch.ones(6, 6, 256)  # All ones
    print(resBlock.forward(ones))

