import torch
import torch.nn as nn
import numpy as np
from torchvision import models
from dcnn import run_node_classification3, A_to_diffusion_kernel
from dcnn import num_hops, num_features
VISUAL_DIM = 512  # number of DOM feature
DSM_DIM = 4*75  # number of DSM feature
HIDDEN_DIM = 100  # feature dimension of hidden layer

# batch_size = 32 #要改
class FusionNet(nn.Module):
    def __init__(self, num_classes=2):
        super(FusionNet, self).__init__()
        net = models.vgg16(pretrained=True)  # Load the VGG16 network parameters from the pre-trained model
        # net = models.resnet50(pretrained=False)
        net.classifier = nn.Sequential()
        self.features = net          # keep feature layer of VGG16
        self.vggclassifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, VISUAL_DIM),
        )
        self.visual = nn.Sequential(
            nn.BatchNorm1d(VISUAL_DIM),
            nn.Linear(VISUAL_DIM, HIDDEN_DIM)
        )
        self.dsm = nn.Sequential(
            nn.BatchNorm1d(DSM_DIM),
            nn.Linear(DSM_DIM, HIDDEN_DIM)
        )
        self.dcnn_net = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU()
        )
        self.myclassifier = nn.Sequential(
            nn.Linear(HIDDEN_DIM, num_classes),
        )
        self.Apow, self.X = self.get_Apow()
        self.activation = nn.Tanh() #ReLU()
        self.gate_activation = nn.Sigmoid() # Sigmoid()  # .LogSigmoid()
        self.W = torch.nn.init.kaiming_uniform_(torch.Tensor(2*HIDDEN_DIM, HIDDEN_DIM))
        self.Wv = torch.nn.init.kaiming_uniform_(torch.Tensor(HIDDEN_DIM, HIDDEN_DIM))
        self.Wt = torch.nn.init.kaiming_uniform_(torch.Tensor(HIDDEN_DIM, HIDDEN_DIM))
        self.W = torch.nn.Parameter(self.W)
        self.Wv = torch.nn.Parameter(self.Wv)
        self.Wt = torch.nn.Parameter(self.Wt)
        self.W_dcnn = torch.nn.init.kaiming_normal_(torch.Tensor(1, num_hops+1, num_features)) #broadcast set 1
        self.W_dcnn = torch.nn.Parameter(self.W_dcnn)

    def forward(self, x, Apow):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.vggclassifier(x)
        vgg_f = self.visual(x)
        Apow_dot_X = torch.matmul(Apow, self.X)
        dsm_fnew = self.dcnn_net(Apow_dot_X)
        dsm_fnew = dsm_fnew.reshape(Apow.shape[0], (num_hops + 1) * num_features)
        dsm_fnew = self.dsm(dsm_fnew)

        hv = torch.mm(vgg_f, self.Wv)
        ht = torch.mm(dsm_fnew, self.Wt)
        feature = torch.cat([vgg_f, dsm_fnew], dim=1)  # concatenate DOM and DSM features
        h = self.activation(torch.cat([hv, ht], dim=1))
        z = self.gate_activation(torch.mm(feature, self.W))
        H = z * h[:, :HIDDEN_DIM] + (1 - z) * h[:, HIDDEN_DIM:]
        out = self.myclassifier(H)

        return out

    def get_Apow(self):
        A, X = run_node_classification3()
        Apow = A_to_diffusion_kernel(A, num_hops)
        Apow = torch.tensor(Apow).cuda()
        X = torch.tensor(X, dtype=torch.float32).cuda()
        return Apow, X
