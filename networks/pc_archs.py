import torch, torch.nn as nn, torch.nn.functional as F
#from torch3d.nn import EdgeConv

# Modified from Torch3D

#class DGCNN(nn.Module):
#    """
#    DGCNN classification architecture from the 
#        `"Dynamic Graph CNN for Learning on Point Clouds" <https://arxiv.org/abs/1801.07829>`_ paper.
#    Args:
#        in_channels (int): Number of channels in the input point set
#        num_classes (int): Number of classes in the dataset
#        dropout (float, optional): Dropout rate in the classifier. Default: 0.5
#    """  # noqa
#
#    def __init__(self, in_channels, num_classes, dropout):
#        super(DGCNN, self).__init__()
#        self.conv1 = EdgeConv(in_channels, 64, 20, bias=False)
#        self.conv2 = EdgeConv(64, 64, 20, bias=False)
#        self.conv3 = EdgeConv(64, 128, 20, bias=False)
#        self.conv4 = EdgeConv(128, 256, 20, bias=False)
#        self.avgpool = nn.AdaptiveAvgPool1d(1)
#        self.maxpool = nn.AdaptiveMaxPool1d(1)
#        self.FD = 512 # 256
#        self.mlp = nn.Sequential(
#            nn.Linear(1024, 512, bias=False),
#            nn.BatchNorm1d(512),
#            nn.LeakyReLU(0.2, True),
#            nn.Dropout(dropout),
#            nn.Linear(512, self.FD, bias=False),
#            nn.BatchNorm1d(self.FD),
#            nn.LeakyReLU(0.2, True),
#            nn.Dropout(dropout),
#        )
#        self.fc = nn.Linear(self.FD, num_classes)
#
#    def forward(self, x): # x in B x np x 3
#        x  = x.transpose(1,2)
#        x1 = self.conv1(x)
#        x2 = self.conv2(x1)
#        x3 = self.conv3(x2)
#        x4 = self.conv4(x3)
#        x = torch.cat([x1, x2, x3, x4], dim=1)
#        x = torch.cat([self.avgpool(x), self.maxpool(x)], dim=1).squeeze(-1)
#        x = self.mlp(x)
#        x = self.fc(x)
#        return x

class VaePointNet(nn.Module):
    def __init__(self, indim, outdim, dropout=0.0):
        super(VaePointNet, self).__init__()
        self.indim  = indim
        self.outdim = outdim
        self.pn     = PointNet(indim, outdim, dropout=dropout)
        self.bn_act = nn.Sequential(
                        nn.BatchNorm1d(outdim),
                        nn.ReLU() )

    def forward(self, x):
        return self.bn_act( self.pn(x) )

class PointNet(nn.Module):
    """
    PointNet classification architecture from the `"PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation" <https://arxiv.org/abs/1612.00593>`_ paper.
    Args:
        in_channels (int): Number of channels in the input point set
        num_classes (int): Number of classes in the dataset
        dropout (float, optional): Dropout rate in the classifier. Default: 0.5
    """  # noqa

    def __init__(self, in_channels, num_classes, dropout=0.5):
        super(PointNet, self).__init__()
        self.final_dim = 512
        self.mlp1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Conv1d(64, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Conv1d(128, 1024, 1, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
        )
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.mlp3 = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Dropout(dropout),
            ####
            nn.Linear(512, self.final_dim, bias=False),
            nn.BatchNorm1d(self.final_dim),
            nn.ReLU(True),
            nn.Dropout(dropout),
        )
        self.fc = nn.Linear(self.final_dim, num_classes)

    def forward(self, x):
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.maxpool(x).squeeze(2)
        x = self.mlp3(x)
        x = self.fc(x)
        return x

class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        B = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        #iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
        #    batchsize, 1)

        iden = torch.eye(3).unsqueeze(0).expand(B, -1, -1).view(B, 9).to(x.device)
        # if x.is_cuda:
        #     iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x

class PointNetWT(nn.Module):
    def __init__(self, in_channels=3, num_classes=1024, global_feat=True):
        super(PointNetWT, self).__init__()
        self.final_dim = num_classes
        self.stn = STN3d() # Spatial transformer
        self.conv1 = torch.nn.Conv1d(in_channels, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        # Global encoding layers
        self.gfc1 = nn.Linear(1024, 1024)
        self.gbn1 = nn.BatchNorm1d(1024)
        self.gfc2 = nn.Linear(1024, self.final_dim)

    def forward(self, x):
        batchsize = x.size()[0] # x = B x C x Np
        n_pts = x.size()[2]
        trans = self.stn(x) # transform
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        #pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))  # x = batch,1024,n(n=2048)
        x = torch.max(x, 2, keepdim=True)[0]  # x = batch,1024,1 # pooling
        x = x.view(-1, 1024)  # x = batch,1024
        # if self.global_feat:
        #     return x, trans
        # else:
        #     x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
        #     return torch.cat([x, pointfeat], 1), trans
        # Global processing
        x = F.relu(self.gbn1(self.gfc1(x)))  # x = batch,final_dim
        x = self.gfc2(x)
        return x







#
