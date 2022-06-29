'''
Created on Nov 10, 2017
Create Model

@author: Lianhai Miao
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import itertools

class GRADI(nn.Module):
    def __init__(self, num_users, num_items, num_groups, embedding_dim, group_member_dict, device, drop_ratio, lmd, eta):
        super(GRADI, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_groups = num_groups
        self.embedding_dim = embedding_dim
        self.group_member_dict = group_member_dict
        self.device = device
        self.drop_ratio = drop_ratio
        self.lmd = lmd
        self.eta = eta

        self.userembeds = nn.Embedding(num_users, self.embedding_dim) 
        self.itemembeds = nn.Embedding(num_items, self.embedding_dim)
        self.groupembeds = nn.Embedding(num_groups, self.embedding_dim)

        self.attention = AttentionLayer(2 * self.embedding_dim, drop_ratio)
       
        self.predictlayer = PredictLayer(3 * self.embedding_dim, drop_ratio)

        self.mlp = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(self.drop_ratio)
        )

        nn.init.xavier_uniform_(self.mlp[0].weight)
        if self.mlp[0].bias is not None:
            self.mlp[0].bias.data.fill_(0.0)


        # initial model
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)
            if isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)

    def forward(self, user_inputs, item_inputs, type_m):
        if type_m == 'group':
            # get the item and group full embedding vectors
            item_embeds_full = self.itemembeds(item_inputs)
            group_embeds_full = self.groupembeds(user_inputs)

            g_embeds_with_attention = torch.zeros([len(user_inputs), self.embedding_dim])

            # start=time.time()(-1) * np.inf *
            user_ids = [self.group_member_dict[usr.item()] for usr in user_inputs] # [B,1]
            MAX_MENBER_SIZE = max([len(menb) for menb in user_ids]) # the great group size = 4
            menb_ids,  group_mask = [None]*len(user_ids),  [None]*len(user_ids)
            for i in range(len(user_ids)):
                postfix = [(-1) * np.inf]*(MAX_MENBER_SIZE - len(user_ids[i]))
                pading_idx = [self.num_users - 1] * (MAX_MENBER_SIZE - len(user_ids[i]))
                menb_ids[i] = user_ids[i] + pading_idx
                group_mask[i] = [0]*len(user_ids[i]) + postfix
            
            menb_ids,  group_mask = torch.Tensor(menb_ids).long().to(self.device),\
                                        torch.Tensor(group_mask).float().to(self.device)
            
            menb_emb =  self.userembeds(menb_ids) # [B, G, C] 

            gro_mem_emb = torch.cat((menb_emb, group_embeds_full.unsqueeze(1).repeat(1, menb_emb.shape[1], 1)), dim=-1) # [4*B, G, 2C]

            attention_out = self.attention(gro_mem_emb) # [4*B, G, 1]

            weight = torch.softmax(attention_out + group_mask.unsqueeze(2), dim=1)  # [4*B, G, 1]
 
            gro_specif_mem_emb = menb_emb + torch.mul(weight, group_embeds_full.unsqueeze(1).repeat(1, menb_emb.shape[1], 1)) # [4*B, G, C]

            ###### Item-specific group representation by bottom-up influence
            men_kdl = self.mlp(gro_specif_mem_emb) # [4*B, G, C]
            item_jd = self.mlp(item_embeds_full)   # [4*B, C]

            self_attention_out = torch.matmul(men_kdl, item_jd.unsqueeze(-1)) / torch.sqrt(torch.tensor(gro_specif_mem_emb.shape[-1]))

            item_specfi_weight = torch.softmax(self_attention_out + group_mask.unsqueeze(2), dim=1)  # [4*B, G, 1]

            g_embeds_with_attention = torch.matmul(item_specfi_weight.transpose(2, 1), gro_specif_mem_emb) 

            g_embeds_with_attention = g_embeds_with_attention.squeeze(1)   # [4*B, C]

            ###### prediction module
            element_embeds = torch.mul(g_embeds_with_attention, item_embeds_full)  # Element-wise product
            new_embeds = torch.cat((element_embeds, g_embeds_with_attention, item_embeds_full), dim=1)
            preds_gro = torch.sigmoid(self.predictlayer(new_embeds))

            return preds_gro

        elif type_m == 'user':
            user_embeds = self.userembeds(user_inputs)
            item_embeds = self.itemembeds(item_inputs)

            element_embeds = torch.mul(user_embeds, item_embeds)  # Element-wise product
            new_embeds = torch.cat((element_embeds, user_embeds, item_embeds), dim=1)
            preds_user = torch.sigmoid(self.predictlayer(new_embeds))

            return preds_user

            
class AttentionLayer(nn.Module):
    """ Attention Layer"""
    def __init__(self, embedding_dim, drop_ratio=0):
        super(AttentionLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, 16),
            nn.ReLU(),
            nn.Dropout(drop_ratio, self.training),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        out = self.linear(x)
        return out
        # weight = F.softmax(out.view(1, -1), dim=1)
        # return weight


class PredictLayer(nn.Module):
    def __init__(self, embedding_dim, drop_ratio=0):
        super(PredictLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, 8),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        out = self.linear(x)
        return out

