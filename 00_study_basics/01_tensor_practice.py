import torch
import torch.nn as nn
import torch.nn.functional as F

print("=== Tensor Shape Practice ===")

# 1. 가상의 환자 데이터 만들기 (Batch=1, Cells=5, Genes=10)
input_data = torch.randn(1, 5, 10)
print(f"1. Input Shape (Bag): {input_data.shape}")

# 2. Linear Layer (유전자 10개 -> 특징 4개로 압축)
layer = nn.Linear(10, 4)
features = layer(input_data)
print(f"2. Features Shape (Encoded): {features.shape}") 

# 3. Attention Layer (특징 4개 -> 중요도 점수 1개)
attn_layer = nn.Linear(4, 1)
scores = attn_layer(features)
print(f"3. Scores Shape (Raw Attention): {scores.shape}")

# 4. Softmax (점수를 확률로 변환)
# dim=1은 Cell 개수(5개)에 대해 합이 1이 되도록 함
alpha = F.softmax(scores, dim=1)
print(f"4. Attention Weights:\n{alpha.squeeze()}")
print(f"   Sum check: {torch.sum(alpha).item()}")