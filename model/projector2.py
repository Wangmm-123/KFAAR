import torch
import torch.nn as nn
import numpy as np
import PIL.Image
# import dnnlib
# import legacy




# class Projector(nn.Module):
#     def __init__(self, ndigit: int = 512):
#         super().__init__()
#         # self.fc1 = nn.Linear(ndigit, 512)

#     def forward(self,  feature: torch.Tensor,k: torch.Tensor):
#         # index_sequence = self.hidden_layers(k)
#         index_sequence = k
#         # 对位置索引序列进行排序
#         sorted_indices = torch.argsort(index_sequence, dim=1)  
#         # 对潜在向量进行置乱和加密
#         W_encrypted = torch.gather(feature, 1, sorted_indices)
        
#         return W_encrypted


# class Projector(nn.Module):
#     def __init__(self, ndigit: int = 8):
#         super().__init__()
#         self.hidden_layers = nn.Sequential(
#             nn.Linear(8, 256),
#             nn.ReLU(),
#             nn.Linear(256, 512),
#             nn.ReLU()
#         )

#     def forward(self, feature: torch.Tensor, k: torch.Tensor):
#         batch_size = feature.size(0)

#         W_encrypted = []
#         for i in range(batch_size):
#             curr_feature = feature[i]
#             curr_k = k[i]

#             index_sequence = self.hidden_layers(curr_k)
#             sorted_indices = torch.argsort(index_sequence)

#             curr_W_encrypted = torch.gather(curr_feature, 0, sorted_indices)

#             W_encrypted.append(curr_W_encrypted)

#         W_encrypted = torch.stack(W_encrypted)

#         return W_encrypted   
    

# class Projector(nn.Module):
#     def __init__(self, ndigit: int = 8) -> None:
#         super().__init__()
#         self.fc1 = nn.Linear(512+ndigit, 512+ndigit)
#         self.fc2 = nn.Linear(512+ndigit, 512)
#
#         self.relu = nn.ReLU()
#
#     def forward(self, feature: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
#         out = torch.cat([feature, k], dim=2)
#
#         out = self.fc1(out)
#         out = self.relu(out)
#         out = self.fc2(out)
#
#         return out

class Projector(nn.Module):
    def __init__(self, ndigit: int = 8) -> None:
        super(Projector, self).__init__()
        self.fc1 = nn.Linear(512+ndigit, 512+ndigit)
        self.fc2 = nn.Linear(512+ndigit, 512)
        self.relu = nn.ReLU()

    def forward(self, feature: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        out = torch.cat([feature, k.unsqueeze(1).expand(-1, feature.size(1), -1)], dim=2)

        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)

        return out



