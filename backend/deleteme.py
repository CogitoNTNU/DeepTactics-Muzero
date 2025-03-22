import torch
import torch.nn.functional as F

v = torch.tensor([0.5044, 0.4956])
t = torch.tensor([0.2043, 0.7957])
print((-t * F.log_softmax(input=v, dim=0)).sum())

import torch

# Predicted probabilities
v = torch.tensor([0.5044, 0.4956])

# Target probabilities
t = torch.tensor([0.2043, 0.7957])

# Compute cross-entropy loss
cross_entropy = -torch.sum(t * torch.log(v))

print("Cross-Entropy Loss:", cross_entropy.item())