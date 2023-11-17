import torch 
from triton_transformers import Transformer 

model = Transformer(
        num_tokens = 256, #vocab size 
        max_seq_len = 1024, 
        dim = 152, 
        depth = 6, 
        heads = 8, 
        dim_head = 64, 
        causual = True, 
        attn_dropout = 0.1, 
        ff_dropout = 0.1, 
        use_triton = True
        ).cuda()

x = torch.randint(0, 20000, (1, 512)).cuda()
labels = torch.randint(0,20000, (1, 512))cuda()

