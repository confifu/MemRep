import torch
from torch import nn
import torch.nn.functional as F
from collections import namedtuple
from torch.nn.utils import clip_grad_norm_

class Memory(nn.Module):
    def __init__(self, numRows, numCols):
        super(Memory, self).__init__()

        self.num_cols = numCols
        self.num_rows = numRows
        self.mem_bias = torch.Tensor().new_full((numRows, numCols), 1e-6)
        
    def init_state(self, batch_size, device):
        self.data =  self.mem_bias.clone().repeat(batch_size, 1, 1).to(device)



class WriteHead(nn.Module):
    
    def __init__(self, memory, hidden_size, fpv, max_shift):
        super(WriteHead, self).__init__()
        self.memory = memory
        self.hidden_size = hidden_size
        self.max_shift = max_shift
        self.mha = nn.MultiheadAttention(embed_dim=self.memory.num_cols, num_heads=4)

        self.fc = nn.Linear(hidden_size,
                            sum(s for s, _ in self.hidden_state_unpacking_scheme()))
        #self.init_params()
        self.write_focus_bias = nn.Parameter(torch.rand(1, fpv, self.memory.num_rows))

    def unpack_hidden_state(self, h):
        chunk_idxs, activations = zip(*self.hidden_state_unpacking_scheme())
        chunks = torch.split(h, chunk_idxs, dim=-1)
        return tuple(activation(chunk) for chunk, activation in zip(chunks, activations))

    def focus_head(self, k, beta, prev_w, g, s, gamma):
        w_c = self._content_weight(k, beta)
        w_g = self._gated_interpolation(w_c, prev_w, g)
        w_s = self._mod_shift(w_g, s)
        w = self._sharpen(w_s, gamma)
        return w

    def _content_weight(self, k, beta):
        '''
        k = k.unsqueeze(1).expand_as(self.memory.data)
        similarity_scores = F.cosine_similarity(k, self.memory.data, dim=2)
        '''
        self.memory.data = self.memory.data.transpose(0, 1)
        k = k.transpose(0, 1)
        _, similarity_scores = self.mha(self.memory.data, k, k)
        self.memory.data = self.memory.data.transpose(0, 1)
        k = k.transpose(0, 1)
        similarity_scores = similarity_scores.transpose(-1, -2)

        #print(similarity_scores.shape, beta.shape)
        w = F.softmax(beta * similarity_scores, dim = -1)
        #print(similarity_scores.shape, w.shape)
        return w
    

    def _gated_interpolation(self, w, prev_w, g):
        return g*w + (1-g)*prev_w

    def _mod_shift(self, w , s):
        unrolled = torch.cat([w[:,:, -self.max_shift:], w, w[:,:, :self.max_shift]], -1)
        return F.conv1d(unrolled.unsqueeze(1), 
                        s.unsqueeze(1))[range(self.batch_size), range(self.batch_size)]
    
    def _sharpen(self, w, gamma):
        w = w.pow(gamma)
        return torch.div(w, w.sum(-1).unsqueeze(-1) + 1e-16)

    def hidden_state_unpacking_scheme(self):
        return [
            # size, activation-function
            (self.memory.num_cols, torch.tanh),                    # k
            (1,                    F.softplus),                    # β 
            (1,                    torch.sigmoid),                 # g
            (2*self.max_shift+1,   lambda x: F.softmax(x, dim=1)), # s
            (1,                    lambda x: F.softplus(x) + 1),   # γ
            (self.memory.num_cols, torch.sigmoid),                 # e
            (self.memory.num_cols, torch.tanh)                     # a
        ] 
        
    
    def write(self, w, e, a):

        if w.dim == 2:
            w = w.unsqueeze(1)
            e = e.unsqueeze(1)
            a = a.unsqueeze(1)
        w = w.transpose(-1, -2)

        memory_erased = self.memory.data * torch.einsum('bmh,bhe->bme', (w, e))
        self.memory.data = memory_erased + torch.einsum('bmh,bhe->bme', (w, a))


    def forward(self, h, prev_w):
        k, beta, g, s, gamma, e, a = self.unpack_hidden_state(self.fc(h))
        w = self.focus_head(k, beta, prev_w, g, s, gamma)

        self.write(w, e, a)
        return w

    def init_state(self, batch_size, device):
        self.batch_size = batch_size
        write_focus = self.write_focus_bias.clone().repeat(batch_size, 1, 1).to(device)
        return torch.softmax(write_focus, dim=1)
