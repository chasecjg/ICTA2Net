import torch
import torch.nn as nn


class CBR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(CBR, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class MMFE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MMFE, self).__init__()
        # 初始的Conv1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # 五个分支
        self.cbr1_1 = CBR(out_channels, out_channels, 1)
        self.cbr1_2 = CBR(out_channels, out_channels, 1)
        self.cbr3 = CBR(out_channels, out_channels, 3, padding=1)
        self.cbr5 = CBR(out_channels, out_channels, 5, padding=2)
        self.cbr7 = CBR(out_channels, out_channels, 7, padding=3)
        # 全局平均池化
        self.gap = nn.AdaptiveAvgPool2d(1)
        # 后续的CBR1
        self.cbr1_post_gap = CBR(out_channels * 3, out_channels, 1)
        # self.cbr1_1_post_gap = CBR(out_channels, in_channels, 1)
        # Sigmoid层
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        x = self.conv1(x)
        branch1 = self.cbr1_1(x)
        branch2 = self.cbr1_2(x)
        branch3 = self.cbr3(x)
        branch4 = self.cbr5(x)
        branch5 = self.cbr7(x)

        branch3 = branch2 + branch3
        branch4 = branch3 + branch4
        branch5 = branch4 + branch5

        x = torch.cat([branch3, branch4, branch5], dim=1)
        x = self.gap(x)
        x = self.cbr1_post_gap(x)

        x = x + branch1
        # x = self.cbr1_1_post_gap(x)
        x = self.sigmoid(x)
        return x
    

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class RF(nn.Module):
    # Revised from: Receptive Field Block Net for Accurate and Fast Object Detection, 2018, ECCV
    # GitHub: https://github.com/ruinmessi/RFBNet
    def __init__(self, in_channel, out_channel):
        super(RF, self).__init__()
        self.relu = nn.ReLU(True)

        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )

        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), dim=1))

        x = self.relu(x_cat + self.conv_res(x))
        return x
# 测试代码
if __name__ == "__main__":
    input_tensor = torch.randn(2, 3, 224, 224)  # 假设输入图像尺寸为224x224，batch size为1，通道数为3
    model = MMFE(3,3)
    output = model(input_tensor)
    output = model(output)
    print(output.shape)