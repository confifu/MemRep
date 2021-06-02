import torch
from torch import nn
import torch.nn.functional as F
from collections import namedtuple
from torch.nn.utils import clip_grad_norm_

class Memory(nn.Module):
    def __init__(self, numRows, numCols):
        super(Memory, self).__init__()

        self.numCols = numCols
        self.numRows = numRows
        self.mem_bias = torch.Tensor().new_full((numRows, numCols), 1e-6)
        
    def init_state(self, batch_size, device):
        self.data =  self.mem_bias.clone().repeat(batch_size, 1, 1).to(device)

class HeadBase(nn.Module):
    
    def __init__(self, memory, hidden_size, max_shift):
        super(HeadBase, self).__init__()
        self.memory = memory
        self.hidden_size = hidden_size
        self.max_shift = max_shift

        self.fc = nn.Linear(hidden_size,
                            sum(s for s, _ in self.hidden_state_unpacking_scheme()))
        self.init_params()

    def forward(self, h):
        raise NotImplementedError

    def hidden_state_unpacking_scheme():
        raise NotImplementedError
           
    def unpack_hidden_state(self, h):
        chunk_idxs, activations = zip(*self.hidden_state_unpacking_scheme())
        chunks = torch.split(h, chunk_idxs, dim=1)
        return tuple(activation(chunk) for chunk, activation in zip(chunks, activations))

    def focus_head(self, k, beta, prev_w, g, s, gamma):
        w_c = self._content_weight(k, beta)
        w_g = self._gated_interpolation(w_c, prev_w, g)
        w_s = self._mod_shift(w_g, s)
        w = self._sharpen(w_s, gamma)
        return w

    def _content_weight(self, k, beta):
        k = k.unsqueeze(1).expand_as(self.memory.data)
        similarity_scores = F.cosine_similarity(k, self.memory.data, dim=2)
        w = F.softmax(beta * similarity_scores, dim = 1)
        return w
    

    def _gated_interpolation(self, w, prev_w, g):
        return g*w + (1-g)*prev_w

    def _mod_shift(self, w , s):
        unrolled = torch.cat([w[:, -self.max_shift:], w, w[:, :self.max_shift]], 1)
        return F.conv1d(unrolled.unsqueeze(1), 
                        s.unsqueeze(1))[range(self.batch_size), range(self.batch_size)]
    
    def _sharpen(self, w, gamma):
        w = w.pow(gamma)
        return torch.div(w, w.sum(1).view(-1, 1) + 1e-16)


    def init_state(self, batch_size):
        self.batch_size = batch_size

    def init_params(self):
        pass


class ReadHead(HeadBase):

    def __init__(self, memory, hidden_sz, max_shift):
        super(ReadHead, self).__init__(memory, hidden_sz, max_shift)

        
    def hidden_state_unpacking_scheme(self):
      return [
            # size, activation-function
            (self.memory.num_cols, torch.tanh),                    # k
            (1,                    F.softplus),                    # β
            (1,                    torch.sigmoid),                 # g
            (2*self.max_shift+1,   lambda x: F.softmax(x, dim=1)), # s
            (1,                    lambda x: 1 + F.softplus(x))    # γ
        ]
        
    def read(self, w):
        return torch.matmul(w.unsqueeze(1), self.memory.data).squeeze(1)

    def forward(self, h, prev_w):
        k, beta, g, s, gamma = self.unpack_hidden_state(self.fc(h))
        w = self.focus_head(k, beta, prev_w, g, s, gamma)
        read = self.read(w)
        return read, w
    
    def init_state(self, batch_size, device):
        self.batch_size = batch_size
        reads = torch.zeros(batch_size, self.memory.num_cols).to(device)
        read_focus = torch.zeros(batch_size, self.memory.num_cols).to(device)
        read_focus[:, 0] = 1.
        return reads, read_focus


class WriteHead(HeadBase):

    def __init__(self, memory, hidden_sz, max_shift):
        super(WriteHead, self).__init__(memory, hidden_sz, max_shift)

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
        
    def erase(self, w, e):
        return self.memory.data * (1 - w.unsqueeze(2) * e.unsqueeze(1))
    
    def write(self, w, e, a):
        memory_erased = self.erase(w, e)
        self.memory.data = memory_erased + (w.unsqueeze(2) * a.unsqueeze(1))

    def forward(self, h, prev_w):
        k, beta, g, s, gamma, e, a = self.unpack_hidden_state(self.fc(h))
        w = self.focus_head(k, beta, prev_w, g, s, gamma)
        self.write(w, e, a)
        return w

    def init_state(self, batch_size, device):
        self.batch_size = batch_size
        write_focus = torch.zeros(batch_size, self.memory.num_rows).to(device)
        write_focus[:, 0] = 1.
        return write_focus