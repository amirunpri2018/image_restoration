import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1,4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.Dropout2d(p=.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(4,8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.Dropout2d(p=.2),

            nn.ReflectionPad2d(1),
            nn.conv2d(8,8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.Dropout2d(p=.2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(8*100*100, 500),  # 8 is O/p of Conv, 100*100 are dimensions of image
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.Dropout2d(p=.2)
        )

    def forward_once(self,x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output
    
    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

class Loss(nn.Moudle):
    def __init__(self, margin=2.0):
        super(Loss, self).__init__()
        self.margin = margin
    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label)*torch.pow(euclidean_distance, 2) + 
                            (label)*torch.pow(torch.clamp(self.margin - 
                            euclidean_distance, min=0.0),2))
        return loss_contrastive

cuda_avail = torch.cuda.is_available()

if cuda_avail:
    net = Network().cuda()
else:
    net = Network()
