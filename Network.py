import torch.nn as nn
import torch.nn.init
from torchvision.models import resnet18, resnet34

resnet = resnet18(pretrained=True)

"""Define Model"""


class CRNN(nn.Module):

    def __init__(self, num_chars, rnn_hidden_size=256, dropout=0.1):
        super(CRNN, self).__init__()
        self.num_chars = num_chars
        self.rnn_hidden_size = rnn_hidden_size
        self.dropout = dropout

        # CNN Part 1
        resnet_modules = list(resnet.children())[:-3]
        # last sequential, 3 basic-blocks cut
        # print("list(resnet.children()): ", list(resnet.children()))
        # print("resnet_modules: ", resnet_modules)
        self.cnn1 = nn.Sequential(*resnet_modules)

        # CNN Part 2

        self.cnn2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 6), stride=3, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        """
        self.cnn2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 6), stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )"""
        self.linear1 = nn.Linear(1536, 256)
        #self.linear1 = nn.Linear(2048, 256)
        # RNN
        self.rnn1 = nn.GRU(input_size=rnn_hidden_size,
                           hidden_size=rnn_hidden_size,
                           bidirectional=True,
                           batch_first=True)
        self.rnn2 = nn.GRU(input_size=rnn_hidden_size,
                           hidden_size=rnn_hidden_size,
                           bidirectional=True,
                           batch_first=True)
        self.linear2 = nn.Linear(self.rnn_hidden_size * 2, num_chars)

    def forward(self, batch):
        batch = self.cnn1(batch)
        #print(batch.size())
        # torch.Size([batch_size, 256, 8, 8])

        batch = self.cnn2(batch)  # [batch_size, channels, height, width]
        #print(batch.size())
        # torch.Size([batch_size, 256, 8, 5])

        batch = batch.permute(0, 3, 1, 2)  # [batch_size, width, channels, height]
        #print(batch.size())
        # torch.Size([batch_size, 5, 256, 8])

        batch_size = batch.size(0)
        T = batch.size(1)  # width
        batch = batch.view(batch_size, T, -1)  # [batch_size, T==width, num_features==channels*height]
        #print(batch.size())
        # torch.Size([batch_size, 5, 2048])

        batch = self.linear1(batch)
        #print(batch.size())
        # torch.Size([batch_size, 5, 256])

        batch, hidden = self.rnn1(batch)
        feature_size = batch.size(2)
        batch = batch[:, :, :feature_size // 2] + batch[:, :, feature_size // 2:]
        #print(batch.size())
        # torch.Size([batch_size, 5, 256])

        batch, hidden = self.rnn2(batch)
        # print(batch.size())
        # torch.Size([batch_size, 5, 512])

        batch = self.linear2(batch)
        # print(batch.size())
        # torch.Size([batch_size, 5, 18])

        batch = batch.permute(1, 0, 2)  # [T==5, batch_size, num_classes==num_features]
        # print(batch.size())
        # torch.Size([5, batch_size, 18])

        return batch


def weights_init(m):
    classname = m.__class__.__name__
    if type(m) in [nn.Linear, nn.Conv2d, nn.Conv1d]:
        torch.nn.init.xavier_uniform_(m.weight) # weight initialization
        if m.bias is not None:
            m.bias.data.fill_(0.01) # initializes the bias to 0.01
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02) # initialize the weight to mean 1.0, deviation 0.02
        m.bias.data.fill_(0) # initializes the bias to 0
