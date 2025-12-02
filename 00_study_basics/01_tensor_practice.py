import torch
import torch.nn as nn
import torch.nn.functional as F

x = torch.randn(1, 3, 8)
enc = nn.Linear(8, 5)
attn = nn.Linear(5, 1)

features = enc(x)
scores = attn(features)
prob = F.softmax(scores, dim=1)

print(prob)