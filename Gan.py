import torch.nn as nn
import torch
import torch.nn.functional as F
class discriminator_dc(nn.Module):
    def __init__(self,dim=0):
        super(discriminator_dc, self).__init__()
        self.linear_1 = torch.nn.Linear(dim,256)
        self.linear_2 = torch.nn.Linear(256, 128)
        self.linear_3 = torch.nn.Linear(128, 32)
        self.linear_4 = torch.nn.Linear(32, 1)
        self.linear_5 = torch.nn.Linear(32, 1)
    def forward(self, input):
        hidden = self.linear_1(input)
        hidden = self.linear_2(hidden)
        hidden = self.linear_3(hidden)
        valid = torch.sigmoid(self.linear_4(hidden))
        label = torch.sigmoid(self.linear_5(hidden))
        # output = output.squeeze(-1)
        return valid,label


class discriminator(nn.Module):
    def __init__(self,dim=0):
        super(discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(dim, 256),  # 用线性变换将输入映射到256维
            nn.LeakyReLU(True),
            nn.Linear(256, 128),  # 线性变换
            nn.LeakyReLU(True),
            nn.Linear(128, 32),  # 线性变换
            nn.LeakyReLU(True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, input):
        output = self.dis(input)
        return output
class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(100, 128),  # 用线性变换将输入映射到256维
            nn.ReLU(True),  # relu激活
            nn.Linear(128, 256),  # 线性变换
            nn.ReLU(True),  # relu激活
            nn.Linear(256, 512),  # 线性变换
            nn.Tanh()  # Tanh激活使得生成数据分布在【-1,1】之间，因为输入的真实数据的经过transforms之后也是这个分布
        )
 
    def forward(self, x):
        x = self.gen(x)
        # x = x.squeeze(-1)
        return x
