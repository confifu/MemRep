import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import math

from .NTM import Memory, WriteHead

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

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x, offset=0):
        x = x + self.pe[offset:offset + x.size(0), :]
        x = self.dropout(x)
        return x


class TransEncoder(nn.Module):
    def __init__(self, d_model, n_head, dim_ff, dropout=0.0, num_layers = 1):
        super(TransEncoder, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model = d_model,
                                                    nhead = n_head,
                                                    dim_feedforward = dim_ff,
                                                    dropout = dropout,
                                                    activation = 'relu')
        encoder_norm = nn.LayerNorm(d_model)
        self.trans_encoder = nn.TransformerEncoder(encoder_layer, num_layers, encoder_norm)

    def forward(self, src, offset=0):
        src = self.pos_encoder(src, offset)
        e_op = self.trans_encoder(src)
        return e_op


class RedundantAttention(nn.Module):
    def __init__(self, device, embedSize, framePerVid, dropout=0.1):
        super(RedundantAttention, self).__init__()
        self.device = device

        self.mha_sim = nn.MultiheadAttention(embed_dim=embedSize, num_heads=4)
        self.fc_attn = nn.Linear(2*framePerVid, framePerVid)
        self.fc1 = nn.Linear(2*embedSize, embedSize)
        self.fc2 = nn.Linear(embedSize, embedSize)

        self.norm1 = nn.LayerNorm(embedSize)
        self.norm2 = nn.LayerNorm(embedSize)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.fc_diff1 = nn.ModuleList([nn.Linear(embedSize, embedSize//4),
                                nn.Dropout(dropout),
                                nn.Linear(embedSize//4, 1)
                            ])

        self.fc_diff2 = nn.ModuleList([nn.Linear(embedSize, embedSize//4),
                                nn.Dropout(dropout),
                                nn.Linear(embedSize//4, 1)
                            ])

    def forward(self, x, offset = 0):
        '''(N, S, E) --> (N, S, E), (N, S, S)'''
        f = x.shape[1]

        I = torch.ones(f).to(self.device)
        xr = torch.einsum('bfe,h->bhfe', (x, I))   #[x, x, x, x ....]  =>  xr[:,0,:,:] == x
        xc = torch.einsum('bfe,h->bfhe', (x, I))   #[x x x x ....]     =>  xc[:,:,0,:] == x
        diff = xr - xc
        x1_attn = torch.einsum('bfge,bfge->bfg', (diff, diff))
        x1_attn = F.softmax(-x1_attn/13.544, dim = -1)
        x1 = torch.bmm(x1_attn, x)

        x2 = x.transpose(0, 1)
        x2, x2_attn = self.mha_sim(x2, x2, x2)
        x2 = x2.transpose(0, 1)

        x_res = torch.cat([x1, x2], dim = -1)

        x = x + self.dropout1(F.relu(self.fc1(x_res)))
        x = self.norm1(x)
        x = x + self.dropout2(F.relu(self.fc2(x)))
        x = self.norm2(x)

        attn_wts = [x1_attn.unsqueeze(1), x2_attn.unsqueeze(1)]

        diff1 = torch.einsum('bfge,bfg->bfge', (diff, x1_attn))
        for layer in self.fc_diff1:
            diff1 = F.relu(layer(diff1))

        diff2 = torch.einsum('bfge,bfg->bfge', (diff, x2_attn))
        for layer in self.fc_diff2:
            diff2 = F.relu(layer(diff2))
        x_diff = [diff1.squeeze(-1), diff2.squeeze(-1)]

        return x, attn_wts, x_diff

class MemRep(nn.Module):
    def __init__(self, framePerVid, numRA = 2):
        super(MemRep, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #self.device = 'cpu'
        self.framePerVid = framePerVid

        #====Encoder====
        self.backbone = X3DBottom()
        
        self.conv3D = nn.ModuleList()
        for i in range(1, 2):
            self.conv3D.extend([nn.Conv3d(in_channels = 192*i,
                                    out_channels = 128*i,
                                    kernel_size = 3,
                                    padding = (3,0,0),
                                    dilation = (3,1,1)),
                                nn.BatchNorm3d(128*i),
                                nn.MaxPool3d(kernel_size = (1, 4, 4))
            ])

        self.memory = Memory(10, 128)

        self.pos_encoder = PositionalEncoding(128)
        self.ra = nn.ModuleList()
        for i in range(0, numRA):
            self.ra.extend([RedundantAttention(self.device, 128, self.framePerVid)])

        self.conv3x3 = nn.ModuleList(
                        [nn.Conv2d(in_channels = 2*numRA,
                                 out_channels = self.framePerVid,
                                 kernel_size = 3,
                                 padding = 1),
                        nn.Conv2d(in_channels = self.framePerVid,
                                 out_channels = self.framePerVid,
                                 kernel_size = 3,
                                 padding = 1),
                        nn.Conv2d(in_channels = self.framePerVid,
                                 out_channels = self.framePerVid//2,
                                 kernel_size = 3,
                                 padding = 1),
                        nn.BatchNorm2d(self.framePerVid//2),
        ])

        convOutShape = (self.framePerVid + 10)*(self.framePerVid//2)
        self.attnProj = nn.Linear(convOutShape, 128)
        self.diffProj = nn.Linear((self.framePerVid + 10)*2*numRA, 128)

        self.aggProj = nn.Linear(128*3, 128)
        self.transEncoder1 = TransEncoder(d_model=128, n_head=2, dropout = 0.2, dim_ff=128, num_layers = 1)
        self.transEncoder2 = TransEncoder(d_model=128, n_head=2, dropout = 0.2, dim_ff=128, num_layers = 1)

        self.writeHead = WriteHead(self.memory, 128*3, self.framePerVid, 5)

        #period length prediction
        self.fc1 = nn.ModuleList(
            [nn.Linear(128, self.framePerVid//2),
            nn.LayerNorm(self.framePerVid//2),
            nn.Linear(self.framePerVid//2, 1),
            ])

        #periodicity prediction
        self.fc2 = nn.ModuleList(
            [nn.Linear(128, self.framePerVid//2),
            nn.LayerNorm(self.framePerVid//2),
            nn.Linear(self.framePerVid//2, 1),
            ])

    def forward(self, x, index=0):
        batch_size, c, frames, h, w = x.shape
        assert frames == self.framePerVid

        x = self.backbone(x)
        for layer in self.conv3D:
            x = F.relu(layer(x))
        x = x.squeeze(-1).squeeze(-1)
        x = x.transpose(-1, -2)

        x_pos_encoded = self.pos_encoder(x.transpose(0, 1), frames*index).transpose(0, 1)
        x = torch.cat([self.memory.data, x_pos_encoded], dim = 1)

        attn_wts = []
        x_diffs = []
        for layer in self.ra:
            x, attn_wt, diff = layer(x)
            attn_wts.extend(attn_wt)
            x_diffs.extend(diff)

        x_diff = torch.cat(x_diffs, dim = -1)
        x_diff = F.relu(self.diffProj(x_diff))

        attn = torch.cat(attn_wts, dim = 1)
        for layer in self.conv3x3:
            attn = F.relu(layer(attn))
        attn = attn.permute(0, 2, 3, 1)
        attn = attn.reshape(batch_size, attn.shape[1], -1)
        attn = F.relu(self.attnProj(attn))

        x_agg = torch.cat([x, attn, x_diff], dim = -1)
        x_agg = F.relu(self.aggProj(x_agg))
        x_agg = x_agg.transpose(0, 1)

        x1 = self.transEncoder1(x_agg)
        y1 = x1.transpose(0, 1)
        y1 = y1[:, -frames:, :]

        x2 = self.transEncoder1(x_agg)
        y2 = x2.transpose(0, 1)
        y2 = y2[:, -frames:, :]

        h = torch.cat([y1, y2, x_pos_encoded], dim = -1)

        self.write_focus = self.writeHead(h, self.write_focus)

        for layer in self.fc1:
            y1 = F.relu(layer(y1))

        for layer in self.fc2:
            y2 = F.relu(layer(y2))

        return y1, y2

    def init_params(self, batch_size):
        self.write_focus = self.writeHead.init_state(batch_size, self.device)
        self.memory.init_state(batch_size, self.device)
