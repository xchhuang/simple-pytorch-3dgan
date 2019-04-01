import torch
import params

'''

model.py

Define our GAN model

The cube_len is 32x32x32, and the maximum number of feature map is 256, 
so the results may be inconsistent with the paper

'''

class net_G(torch.nn.Module):
    def __init__(self, args):
        super(net_G, self).__init__()
        self.args = args
        self.cube_len = params.cube_len
        self.bias = params.bias
        self.z_dim = params.z_dim
        self.f_dim = 32

        padd = (0, 0, 0)
        if self.cube_len == 32:
            padd = (1,1,1)

        self.layer1 = self.conv_layer(self.z_dim, self.f_dim*8, kernel_size=4, stride=2, padding=padd, bias=self.bias)
        self.layer2 = self.conv_layer(self.f_dim*8, self.f_dim*4, kernel_size=4, stride=2, padding=(1, 1, 1), bias=self.bias)
        self.layer3 = self.conv_layer(self.f_dim*4, self.f_dim*2, kernel_size=4, stride=2, padding=(1, 1, 1), bias=self.bias)
        self.layer4 = self.conv_layer(self.f_dim*2, self.f_dim, kernel_size=4, stride=2, padding=(1, 1, 1), bias=self.bias)
        
        self.layer5 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.f_dim, 1, kernel_size=4, stride=2, bias=self.bias, padding=(1, 1, 1)),
            torch.nn.Sigmoid()
            # torch.nn.Tanh()
        )

    def conv_layer(self, input_dim, output_dim, kernel_size=4, stride=2, padding=(1,1,1), bias=False):
        layer = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(input_dim, output_dim, kernel_size=kernel_size, stride=stride, bias=bias, padding=padding),
            torch.nn.BatchNorm3d(output_dim),
            torch.nn.ReLU(True)
            # torch.nn.LeakyReLU(self.leak_value, True)
        )
        return layer

    def forward(self, x):
        out = x.view(-1, self.z_dim, 1, 1, 1)
        # print(out.size())  # torch.Size([32, 200, 1, 1, 1])
        out = self.layer1(out)
        # print(out.size())  # torch.Size([32, 256, 2, 2, 2])
        out = self.layer2(out)
        # print(out.size())  # torch.Size([32, 128, 4, 4, 4])
        out = self.layer3(out)
        # print(out.size())  # torch.Size([32, 64, 8, 8, 8])
        out = self.layer4(out)
        # print(out.size())  # torch.Size([32, 32, 16, 16, 16])
        out = self.layer5(out)
        # print(out.size())  # torch.Size([32, 1, 32, 32, 32])
        out = torch.squeeze(out)
        return out


class net_D(torch.nn.Module):
    def __init__(self, args):
        super(net_D, self).__init__()
        self.args = args
        self.cube_len = params.cube_len
        self.leak_value = params.leak_value
        self.bias = params.bias

        padd = (0,0,0)
        if self.cube_len == 32:
            padd = (1,1,1)

        self.f_dim = 32

        self.layer1 = self.conv_layer(1, self.f_dim, kernel_size=4, stride=2, padding=(1,1,1), bias=self.bias)
        self.layer2 = self.conv_layer(self.f_dim, self.f_dim*2, kernel_size=4, stride=2, padding=(1,1,1), bias=self.bias)
        self.layer3 = self.conv_layer(self.f_dim*2, self.f_dim*4, kernel_size=4, stride=2, padding=(1,1,1), bias=self.bias)
        self.layer4 = self.conv_layer(self.f_dim*4, self.f_dim*8, kernel_size=4, stride=2, padding=(1,1,1), bias=self.bias)

        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv3d(self.f_dim*8, 1, kernel_size=4, stride=2, bias=self.bias, padding=padd),
            torch.nn.Sigmoid()
        )

        # self.layer5 = torch.nn.Sequential(
        #     torch.nn.Linear(256*2*2*2, 1),
        #     torch.nn.Sigmoid()
        # )

    def conv_layer(self, input_dim, output_dim, kernel_size=4, stride=2, padding=(1,1,1), bias=False):
        layer = torch.nn.Sequential(
            torch.nn.Conv3d(input_dim, output_dim, kernel_size=kernel_size, stride=stride, bias=bias, padding=padding),
            torch.nn.BatchNorm3d(output_dim),
            torch.nn.LeakyReLU(self.leak_value, inplace=True)
        )
        return layer

    def forward(self, x):
        # out = torch.unsqueeze(x, dim=1)
        out = x.view(-1, 1, self.cube_len, self.cube_len, self.cube_len)
        # print(out.size()) # torch.Size([32, 1, 32, 32, 32])
        out = self.layer1(out)
        # print(out.size())  # torch.Size([32, 32, 16, 16, 16])
        out = self.layer2(out)
        # print(out.size())  # torch.Size([32, 64, 8, 8, 8])
        out = self.layer3(out)
        # print(out.size())  # torch.Size([32, 128, 4, 4, 4])
        out = self.layer4(out)
        # print(out.size())  # torch.Size([32, 256, 2, 2, 2])
        # out = out.view(-1, 256*2*2*2)
        # print (out.size())
        out = self.layer5(out)
        # print(out.size())  # torch.Size([32, 1, 1, 1, 1])
        out = torch.squeeze(out)
        return out

