import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F

from NTM import Memory, ReadHead, WriteHead

class X3DBottom(nn.Module):
    def __init__(self):
        super(X3DBottom, self).__init__()
        self.original_model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_s', pretrained=True)

        self.activation = {}
        h = self.original_model.blocks[4].register_forward_hook(self.getActivation('comp'))
        
    def getActivation(self, name):
        def hook(model, input, output):
            self.activation[name] = output
        return hook

    def forward(self, x):
        self.original_model(x)
        output = self.activation['comp']
        return output

class SimScore(nn.Module):
    def __init__(self, device, framePerVid):
        super(SimScore, self).__init__()
        self.device = device

        self.fc1 = nn.Linear(framePerVid, framePerVid//2)
        self.ln1 = nn.LayerNorm(framePerVid//2)
        self.fc2 = nn.Linear(framePerVid//2, 1)

    def forward(self, x):
        '''(N, S, E)  --> (N, S, S) --> (N, S, 1)'''
        f = x.shape[1]
        
        I = torch.ones(f).to(self.device)
        xr = torch.einsum('bfe,h->bhfe', (x, I))   #[x, x, x, x ....]  =>  xr[:,0,:,:] == x
        xc = torch.einsum('bfe,h->bfhe', (x, I))   #[x x x x ....]     =>  xc[:,:,0,:] == x
        diff = xr - xc
        sim = torch.einsum('bfge,bfge->bfg', (diff, diff))
        sim = F.softmax(-sim/13.544, dim = -1)

        simScore = F.relu(self.ln1(self.fc1(sim)))
        simScore = F.relu(self.fc2(simScore))
        return simScore

class MemRep(nn.Module):
    def __init__(self, framePerVid):
        super(MemRep, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.framePerVid = framePerVid

        #====Encoder====
        self.backbone = X3DBottom()
        
        self.conv3D = nn.ModuleList()
        for i in range(1, 2):
            self.conv3D.extend([nn.Conv3d(in_channels = 192*i,
                                    out_channels = 256*i,
                                    kernel_size = 3,
                                    padding = (3,0,0),
                                    dilation = (3,1,1)),
                                nn.BatchNorm3d(256*i),
                                nn.MaxPool3d(kernel_size = (1, 4, 4))
            ])

        '''
        self.memory = Memory(10, 512)
        self.readHeads = ReadHead(self.memory)
        self.writeHeads = WriteHead(self.memory)

        self.sims = SimScore(self.device, self.framePerVid)
        '''


    def forward(self, x):
        batch_size, c, frames, h, w = x.shape
        assert frames == self.framePerVid

        #x = x.view(-1, c, h, w)
        x = self.backbone(x)

        print(x.shape)
        for layer in self.conv3D:
            x = F.relu(layer(x))
        x = x.squeeze(-1).squeeze(-1)
        print(x.shape)
        return x

'''
        read, readAttn = self.readHeads(x, prevReadAttn)
        writeAttn = self.writeHeads(x, readAttn, prevWriteAttn)
'''        