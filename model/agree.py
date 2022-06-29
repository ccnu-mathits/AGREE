'''
Created on Nov 10, 2017
Create Model

@author: Lianhai Miao
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import itertools

class AGREE(nn.Module):
    def __init__(self, num_users, num_items, num_groups, embedding_dim, group_member_dict, device, drop_ratio, lmd, eta):
        super(AGREE, self).__init__()
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
        self.attention_u = AttentionLayer(self.embedding_dim, drop_ratio)
        self.self_attention = SelfAttentionLayer(2 * self.embedding_dim, drop_ratio)
        self.self_attention_tuser = SelfAttentionLayer_tuser(self.embedding_dim, drop_ratio)

        self.predictlayer = PredictLayer(3 * self.embedding_dim, drop_ratio)
        self.fcl = nn.Linear(self.embedding_dim, 1)
        
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

            # start=time.time()

            user_ids = [self.group_member_dict[usr.item()] for usr in user_inputs] # [B,1]
            MAX_MENBER_SIZE = max([len(menb) for menb in user_ids]) # the great group size = 4
            menb_ids, item_ids, mask = [None]*len(user_ids),  [None]*len(user_ids),  [None]*len(user_ids)
            for i in range(len(user_ids)):
                postfix = [0]*(MAX_MENBER_SIZE - len(user_ids[i]))
                menb_ids[i] = user_ids[i] + postfix
                item_ids[i] = [item_inputs[i].item()]*len(user_ids[i]) + postfix
                mask[i] = [1]*len(user_ids[i]) + postfix
            
            menb_ids, item_ids, mask = torch.Tensor(menb_ids).long().to(self.device),\
                                        torch.Tensor(item_ids).long().to(self.device),\
                                        torch.Tensor(mask).float().to(self.device)
            
            menb_emb =  self.userembeds(menb_ids) # [B, N, C] 
            menb_emb *= mask.unsqueeze(dim=-1) # [B, N, C] 
            item_emb = self.itemembeds(item_ids) # [B, N, C] 
            item_emb *= mask.unsqueeze(dim=-1) # [B, N, C]
            menbs_item_emb = torch.cat((menb_emb, item_emb), dim=-1) # [B, N, 2C], N=MAX_MENBER_SIZE
            # group_item_emb = group_item_emb.view(-1, group_item_emb.size(-1)) # [B * N, 2C]
            attn_weights = self.attention(menbs_item_emb)# [B, N, 1]
            # attn_weights = attn_weights.squeeze(dim=-1) # [B, N]
            attn_weights = torch.clip(attn_weights.squeeze(dim=-1), -50, 50)
            # attn_weights = attn_weights.view(menb_ids.size(0), -1) # [B, N] 
            attn_weights_exp = attn_weights.exp() * mask # [B, N]
            attn_weights_sm = attn_weights_exp/torch.sum(attn_weights_exp, dim=-1, keepdim=True) # [B, N] 
            attn_weights_sm = attn_weights_sm.unsqueeze(dim=1) # [B, 1, N]
            g_embeds_with_attention = torch.bmm(attn_weights_sm, menb_emb) # [B, 1, C]
            g_embeds_with_attention = g_embeds_with_attention.squeeze(dim=1)
            # print(time.time() - start)
            
            # put the g_embeds_with_attention matrix to GPU
            g_embeds_with_attention = g_embeds_with_attention.to(self.device)
            
            # obtain the group embedding which consists of two components: user embedding aggregation and group preference embedding
            g_embeds = self.lmd * g_embeds_with_attention + group_embeds_full     # AGREE
            # g_embeds = g_embeds_with_attention                           # AGREE_U
            # g_embeds = group_embeds_full                                 # AGREE_G
            # g_embeds = torch.add(torch.mul(g_embeds_with_attention, 0.4), torch.mul(group_embeds_full, 0.6))
                
            element_embeds = torch.mul(g_embeds, item_embeds_full)  # Element-wise product
            new_embeds = torch.cat((element_embeds, g_embeds, item_embeds_full), dim=1)
            # preds_gro = torch.sigmoid(self.predictlayer(new_embeds))
            preds_gro = self.predictlayer(new_embeds)

            # element_embeds = torch.mul(g_embeds, item_embeds_full)
            # preds_gro = torch.sigmoid(element_embeds.sum(1))

            return preds_gro

        if type_m == 'group_fixed_agg':
            # get the item and group full embedding vectors
            item_embeds_full = self.itemembeds(item_inputs)
            group_embeds_full = self.groupembeds(user_inputs)

            g_embeds_with_attention = torch.zeros([len(user_inputs), self.embedding_dim])

            user_ids = [self.group_member_dict[usr.item()] for usr in user_inputs]
            MAX_MENBER_SIZE = max([len(menb) for menb in user_ids])
            menb_ids, mask = [None]*len(user_ids), [None]*len(user_ids)
            for i in range(len(user_ids)):
                postfix = [0]*(MAX_MENBER_SIZE - len(user_ids[i]))
                menb_ids[i] = user_ids[i] + postfix
                mask[i] = [1]*len(user_ids[i]) + postfix
            
            menb_ids, mask = torch.Tensor(menb_ids).long().to(self.device),\
                                        torch.Tensor(mask).float().to(self.device)
            
            menb_emb =  self.userembeds(menb_ids)  # [B, N, C]
            menb_emb *= mask.unsqueeze(dim=-1)     # [B, N, C] 

            g_embeds_with_avg = torch.mean(menb_emb, dim=1) # [B,C]
            # g_embeds_with_lm  = torch.min(menb_emb, dim=1).values
            # g_embeds_with_ms = torch.max(menb_emb, dim=1).values
            # g_embeds_with_exp = torch.median(menb_emb, dim=1).values
        
            
            # obtain the group embedding which consists of two components: user embedding aggregation and group preference embedding
            g_embeds = g_embeds_with_avg # NCF-AVG or BPR-AVG
            # g_embeds = g_embeds_with_avg + group_embeds_full
            # g_embeds = g_embeds_with_lm + group_embeds_full
            # g_embeds = g_embeds_with_ms + group_embeds_full
            # g_embeds = g_embeds_with_exp + group_embeds_full
            # g_embeds = torch.add(torch.mul(g_embeds_with_attention, 0.7), torch.mul(group_embeds_full, 0.3))
                
            element_embeds = torch.mul(g_embeds, item_embeds_full)  # Element-wise product
            new_embeds = torch.cat((element_embeds, g_embeds, item_embeds_full), dim=1)
            preds_gro = torch.sigmoid(self.predictlayer(new_embeds))

            return preds_gro

        elif type_m == 'sa_group':
            # get the item and group full embedding vectors
            item_embeds_full = self.itemembeds(item_inputs)   # [B, C]
            group_embeds_full = self.groupembeds(user_inputs) # [B, C]

            g_embeds_with_attention = torch.zeros([len(user_inputs), self.embedding_dim])
            # g_embeds_with_self_attention = torch.zeros([len(user_inputs), self.embedding_dim])

            user_ids = [self.group_member_dict[usr.item()] for usr in user_inputs] # [B,1]
            MAX_MENBER_SIZE = max([len(menb) for menb in user_ids]) # the great group size = 4
            menb_ids, item_ids, mask = [None]*len(user_ids),  [None]*len(user_ids),  [None]*len(user_ids)
            for i in range(len(user_ids)):
                postfix = [0]*(MAX_MENBER_SIZE - len(user_ids[i]))
                menb_ids[i] = user_ids[i] + postfix
                item_ids[i] = [item_inputs[i].item()]*len(user_ids[i]) + postfix
                mask[i] = [1]*len(user_ids[i]) + postfix
            # [B, N]
            menb_ids, item_ids, mask = torch.Tensor(menb_ids).long().to(self.device),\
                                        torch.Tensor(item_ids).long().to(self.device),\
                                        torch.Tensor(mask).float().to(self.device)
            
            
            menb_emb =  self.userembeds(menb_ids) # [B, N, C] 
            menb_emb *= mask.unsqueeze(dim=-1) # [B, N, C] 
            item_emb = self.itemembeds(item_ids) # [B, N, C] 
            item_emb *= mask.unsqueeze(dim=-1) # [B, N, C]

            #######################################
            #### Self-attention part ##############
            #######################################
            proj_query_emb, proj_key_emb, proj_value_emb = self.self_attention_tuser(menb_emb) # [B, N, C/2], [B, N, C/2], [B, N, C]
            proj_query_emb_new = proj_query_emb * mask.unsqueeze(dim=-1)
            proj_key_emb_new = proj_key_emb * mask.unsqueeze(dim=-1)
            energy = torch.bmm(proj_query_emb_new, proj_key_emb_new.permute(0,2,1)) # [B, N , N]

            energy_exp = energy.exp() * mask.unsqueeze(dim=1)

            energy_exp_softmax = energy_exp/torch.sum(energy_exp, dim=-1, keepdim=True) # [B, N, N] 
            
            menb_emb_out = torch.bmm(energy_exp_softmax, proj_value_emb) # [B, N, N] * [B, N, C] = [B, N, C]
            menb_emb_out_new = menb_emb_out * mask.unsqueeze(dim=-1)
            overall_menb_out = menb_emb_out_new + menb_emb # [B, N, C]

            ######################################
            #### Vanilla attention part ########
            ######################################

            # g_embeds_with_self_attention = torch.sum(overall_menb_out, dim=1)/torch.sum(mask, dim=1, keepdim=True) # [B, C]/[B,1] = [B,C]
            group_item_emb = torch.cat((overall_menb_out, item_emb), dim=-1) # [B, N, 2C], N=MAX_MENBER_SIZE
            attn_weights = self.attention(group_item_emb)# [B, N, 1]

            # group_item_emb = overall_menb_out # [B, N, C], N=MAX_MENBER_SIZE
            # attn_weights = self.attention_u(group_item_emb)# [B, N, 1]
            attn_weights = torch.clip(attn_weights.squeeze(dim=-1), -50, 50)
            # attn_weights = attn_weights.squeeze(dim=-1) # [B, N]
            attn_weights_exp = attn_weights.exp() * mask # [B, N]
            attn_weights_sm = attn_weights_exp/torch.sum(attn_weights_exp, dim=-1, keepdim=True) # [B, N] 
            attn_weights_sm = attn_weights_sm.unsqueeze(dim=1) # [B, 1, N]
            g_embeds_with_attention = torch.bmm(attn_weights_sm, overall_menb_out) # [B, 1, C]
            g_embeds_with_attention = g_embeds_with_attention.squeeze(dim=1)

            # put the g_embeds_with_attention matrix to GPU
            g_embeds_with_attention = g_embeds_with_attention.to(self.device)
            # g_embeds_with_self_attention = g_embeds_with_self_attention.to(self.device)

            # obtain the group embedding which consists of two components: user embedding aggregation and group preference embedding
            g_embeds = self.lmd * g_embeds_with_attention + group_embeds_full
            # g_embeds = g_embeds_with_attention # SAGREE_U
            # g_embeds = group_embeds_full         # SAGREE_G
            # g_embeds = torch.add(torch.mul(g_embeds_with_attention, 0.7), torch.mul(group_embeds_full, 0.3))
                
            element_embeds = torch.mul(g_embeds, item_embeds_full)  # Element-wise product
            preds_gro = torch.sigmoid(self.fcl(element_embeds))
            # new_embeds = torch.cat((element_embeds, g_embeds, item_embeds_full), dim=1)
            # preds_gro = torch.sigmoid(self.predictlayer(new_embeds))

            return preds_gro
        
        elif type_m == 'H-fixed-agg-GR':
            # get the item and group full embedding vectors
            item_embeds_full = self.itemembeds(item_inputs)   # [B, C]
            group_embeds_full = self.groupembeds(user_inputs) # [B, C]
            g_embeds_with_attention = torch.zeros([len(user_inputs), self.embedding_dim])
            
            user_ids = [self.group_member_dict[usr.item()] for usr in user_inputs] # [B,1]
            MAX_MENBER_SIZE = max([len(menb) for menb in user_ids]) # the great group size = 4
            menb_ids, item_ids, mask = [None]*len(user_ids),  [None]*len(user_ids),  [None]*len(user_ids)
            for i in range(len(user_ids)):
                postfix = [0]*(MAX_MENBER_SIZE - len(user_ids[i]))
                menb_ids[i] = user_ids[i] + postfix
                item_ids[i] = [item_inputs[i].item()]*len(user_ids[i]) + postfix
                mask[i] = [1]*len(user_ids[i]) + postfix
            # [B, N]
            menb_ids, item_ids, mask = torch.Tensor(menb_ids).long().to(self.device),\
                                        torch.Tensor(item_ids).long().to(self.device),\
                                        torch.Tensor(mask).float().to(self.device)
            
            menb_emb =  self.userembeds(menb_ids) # [B, N, C] 
            menb_emb *= mask.unsqueeze(dim=-1) # [B, N, C] 
            item_emb = self.itemembeds(item_ids) # [B, N, C] 
            item_emb *= mask.unsqueeze(dim=-1) # [B, N, C]

            #######################################
            #### Self-attention part ##############
            #######################################
            proj_query_emb, proj_key_emb, proj_value_emb = self.self_attention_tuser(menb_emb) # [B, N, C/2], [B, N, C/2], [B, N, C]
            proj_query_emb_new = proj_query_emb * mask.unsqueeze(dim=-1)
            proj_key_emb_new = proj_key_emb * mask.unsqueeze(dim=-1)
            energy = torch.bmm(proj_query_emb_new, proj_key_emb_new.permute(0,2,1)) # [B, N , N]

            energy_exp = energy.exp() * mask.unsqueeze(dim=1)

            energy_exp_softmax = energy_exp/torch.sum(energy_exp, dim=-1, keepdim=True) # [B, N, N] 
            
            menb_emb_out = torch.bmm(energy_exp_softmax, proj_value_emb) # [B, N, N] * [B, N, C] = [B, N, C]
            menb_emb_out_new = menb_emb_out * mask.unsqueeze(dim=-1)
            overall_menb_out = menb_emb_out_new + menb_emb # [B, N, C]

            ######################################
            #### fixed aggregation strategy part ########
            ######################################
            # g_embeds_with_avg = torch.mean(overall_menb_out, dim=1) # [B,C]
            # g_embeds_with_lm  = torch.min(menb_emb, dim=1).values
            # g_embeds_with_ms = torch.max(menb_emb, dim=1).values
            g_embeds_with_exp = torch.median(menb_emb, dim=1).values
        
            
            # obtain the group embedding which consists of two components: user embedding aggregation and group preference embedding
            # g_embeds = g_embeds_with_avg + group_embeds_full
            # g_embeds = g_embeds_with_lm + group_embeds_full
            # g_embeds = g_embeds_with_ms + group_embeds_full
            g_embeds = g_embeds_with_exp + group_embeds_full
            # g_embeds = torch.add(torch.mul(g_embeds_with_attention, 0.7), torch.mul(group_embeds_full, 0.3))

            element_embeds = torch.mul(g_embeds, item_embeds_full)  # Element-wise product
            new_embeds = torch.cat((element_embeds, g_embeds, item_embeds_full), dim=1)
            preds_gro = torch.sigmoid(self.predictlayer(new_embeds))

            # preds_gro = torch.sigmoid(self.fcl(element_embeds))

            return preds_gro
        
        elif type_m == 'target_user_HA':
            # get the target user and item embedding vectors
            user_embeds = self.userembeds(user_inputs)
            item_embeds = self.itemembeds(item_inputs)
        
            # start=time.time()
            # get the group id (key) of the user_inputs and then get all the group member ids(group_user_ids) in the group
            # Get each user_inputs' keys(groups) in the self.group_menb_dict, Note! one user_input may belong to more than one group!!!
            user_inputs_keys = [self.get_keys(self.group_member_dict, usr.item()) for usr in user_inputs]

            new_user_inputs = [None] * len(user_inputs)
            new_item_inputs = [None] * len(user_inputs)
            for i in range(len(user_inputs)):
                new_user_inputs[i] = [user_inputs[i]] * len(user_inputs_keys[i]) 
                new_item_inputs[i] = [item_inputs[i]] * len(user_inputs_keys[i])

            new_user_inputs = [usr for u in new_user_inputs for usr in u] # flatten the nested list new_user_inputs  length = X
            new_user_inputs = torch.Tensor(new_user_inputs).long().to(self.device) # shape: (X,)
            new_user_embeds = self.userembeds(new_user_inputs) # shape:[X, C]

            new_item_inputs = [item for i in new_item_inputs for item in i]
            # itertools(itertools.chain(*new_item_inputs)) # another method to flatten the nested list new_item_inputs
            new_item_inputs = torch.Tensor(new_item_inputs).long().to(self.device) # shape: (X,)
            new_item_embeds = self.itemembeds(new_item_inputs) # shape: [X, C]


            group_input = [group_id for group in user_inputs_keys for group_id in group] # flatten the user_inputs_keys which is a nested list
            group_user_ids = [self.group_member_dict[k] for k in group_input] # length = X

            # get the biggest group size
            MAX_MENBER_SIZE = max([len(menb) for menb in group_user_ids]) # the great group size = 4
            # menb_ids is group members and empty members, mask1 is to mask the empty members, mask is to mask all the other members that is not the user_input id
            menb_ids, mask1 = [None]*len(group_user_ids),  [None]*len(group_user_ids)
            for i in range(len(group_user_ids)):
                postfix = [0]*(MAX_MENBER_SIZE - len(group_user_ids[i])) 
                menb_ids[i] = group_user_ids[i] + postfix

                mask1[i] = [1]*len(group_user_ids[i]) + postfix

            menb_ids, mask1 = torch.Tensor(menb_ids).long().to(self.device),\
                                        torch.Tensor(mask1).float().to(self.device)
            # [X,N] : menb_ids, mask1
            
            mask = (menb_ids == new_user_inputs.unsqueeze(-1)).float().to(self.device) # [X, N]
            # Get the menb_emb
            menb_emb =  self.userembeds(menb_ids) # [X, N, C] 
            menb_emb *= mask1.unsqueeze(dim=-1) # [X, N, C] * [X,N,1] = [X,N,C] Turn the empty menber rows into empty rows
            
            ## Self-attention part #########
            ################################
            proj_query_emb, proj_key_emb, proj_value_emb = self.self_attention_tuser(menb_emb) # [X, N, C/2], [X, N, C/2], [X, N, C]
            proj_query_emb_new = proj_query_emb * mask1.unsqueeze(dim=-1)
            proj_key_emb_new = proj_key_emb * mask1.unsqueeze(dim=-1)
            energy = torch.bmm(proj_query_emb_new, proj_key_emb_new.permute(0,2,1)) # [X, N , N]

            energy_exp = energy.exp() * mask1.unsqueeze(dim=1)

            energy_exp_softmax = energy_exp/torch.sum(energy_exp, dim=-1, keepdim=True) # [X, N, N] 
            
            menb_emb_out = torch.bmm(energy_exp_softmax, proj_value_emb) # [X, N, N] * [X, N, C] = [X, N, C]
            menb_emb_out_new = menb_emb_out * mask1.unsqueeze(dim=-1) # [X,N,C]
            user_emb_out = menb_emb_out_new * mask.unsqueeze(-1) # [X,N,C] * [X,N,1] = [X,N,C]
            user_emb_out_new = torch.sum(user_emb_out, 1) # collapse the rows of user_emb_out and get a [X, C] matrix
            overall_user_emb_out = user_emb_out_new + new_user_embeds # shape: [X, C]
            # overall_user_emb_out = user_emb_out_new # shape: [X, C]

            #############################
            ## Vanilla attention part ###
            #############################
            
            attn_weights = self.attention_u(overall_user_emb_out)# [X, 1]

            # Get a mask matrix to detect which user has joined more than one group
            mask2 = [None] * len(user_inputs)
            for i in range(len(user_inputs)):
                mask2[i] = (new_user_inputs == user_inputs[i]).cpu().numpy()

            mask2 = torch.Tensor(mask2).float().to(self.device) # The shape of mask2: [B, X]
            # multiplies each element of mask2 with the corresponding element of the attn_weights
            new_mask2 = torch.mul(mask2, attn_weights.view(1, -1).exp()) #[B, X]
            # softmax
            new_mask2_sm = new_mask2/torch.sum(new_mask2, dim=-1, keepdim=True)
            # get each user's integrated preference
            new_overall_user_emb_out = torch.mm(new_mask2_sm, overall_user_emb_out) # [B, X] * [X, C] = [B, C]
            # combine the user generated preference with its dedicate embedding: user_embeds
            overall_menb_out = self.eta * new_overall_user_emb_out + user_embeds # [B, C]

            # get dot product between user preference and item preference vectors
            element_embeds = torch.mul(overall_menb_out, item_embeds)  # Element-wise product
            element_embeds = element_embeds.to(self.device)
            # pooling layer
            new_embeds = torch.cat((element_embeds, overall_menb_out, item_embeds), dim=1)
            # rating prediction
            preds_r = torch.sigmoid(self.predictlayer(new_embeds))

            return preds_r

            
        elif type_m == 'user':
            user_embeds = self.userembeds(user_inputs)
            item_embeds = self.itemembeds(item_inputs)

            # element_embeds = torch.mul(user_embeds, item_embeds)  # Element-wise product
            # new_embeds = torch.cat((element_embeds, user_embeds, item_embeds), dim=1)
            # # preds_user = torch.sigmoid(self.predictlayer(new_embeds))
            # preds_user = self.predictlayer(new_embeds)
            element_embeds = torch.mul(user_embeds, item_embeds)  # Element-wise product
            preds_user = torch.sigmoid(element_embeds.sum(1))

            return preds_user

            
    def get_keys(self, d, value):

        return [k for k, v in d.items() if value in v]


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

class SelfAttentionLayer(nn.Module):
    """ Self attention Layer"""
    def __init__(self, embedding_dim, drop_ratio=0.1):
        super(SelfAttentionLayer, self).__init__()
        self.embedding_dim = embedding_dim

        self.query_linear = nn.Sequential()
        self.query_linear.add_module('fc_ise1_query', nn.Linear(embedding_dim, embedding_dim//2))
        self.query_linear.add_module('ac_ise1_query', nn.ReLU(True))
        self.query_linear.add_module('dropout_query', nn.Dropout(drop_ratio))


        self.key_linear = nn.Sequential()
        self.key_linear.add_module('fc_ise1_key', nn.Linear(embedding_dim, embedding_dim//2))
        self.key_linear.add_module('ac_ise1_key', nn.ReLU(True))
        self.query_linear.add_module('dropout_key', nn.Dropout(drop_ratio))

        self.value_linear = nn.Sequential()
        self.value_linear.add_module('fc_ise1_value', nn.Linear(embedding_dim, embedding_dim//2))
        self.value_linear.add_module('ac_ise1_value', nn.ReLU(True))
        self.query_linear.add_module('dropout_value', nn.Dropout(drop_ratio))

    def forward(self, x):
        """ 
            Inputs :
                x  : a group members' embeddings cat item embeddings [B, N, 2C]
            Returns :
                out : out : self attention value + input feature         
        """
        proj_query = self.query_linear(x) # [B, N , C]
        proj_key = self.key_linear(x) # [B, N , C]
        proj_value = self.value_linear(x) # [B, N , C]
        
        return proj_query, proj_key, proj_value

class SelfAttentionLayer_tuser(nn.Module):
    """ Self attention Layer"""
    def __init__(self, embedding_dim, drop_ratio=0.1):
        super(SelfAttentionLayer_tuser, self).__init__()
        self.embedding_dim = embedding_dim

        self.query_linear = nn.Sequential()
        self.query_linear.add_module('fc_ise1_query', nn.Linear(embedding_dim, embedding_dim//2))
        self.query_linear.add_module('ac_ise1_query', nn.ReLU(True))
        self.query_linear.add_module('dropout_query', nn.Dropout(drop_ratio))

        self.key_linear = nn.Sequential()
        self.key_linear.add_module('fc_ise1_key', nn.Linear(embedding_dim, embedding_dim//2))
        self.key_linear.add_module('ac_ise1_key', nn.ReLU(True))
        self.key_linear.add_module('dropout_key', nn.Dropout(drop_ratio))

        self.value_linear = nn.Sequential()
        self.value_linear.add_module('fc_ise1_value', nn.Linear(embedding_dim, embedding_dim))
        self.value_linear.add_module('ac_ise1_value', nn.ReLU(True))
        self.value_linear.add_module('value_query', nn.Dropout(drop_ratio))

    def forward(self, x):
        """ 
            Inputs :
                x  : a group members' embeddings cat item embeddings [B, N, 2C]
            Returns :
                out : out : self attention value + input feature         
        """
        proj_query = self.query_linear(x) # [B, N , C]
        proj_key = self.key_linear(x) # [B, N , C]
        proj_value = self.value_linear(x) # [B, N , C]
        
        return proj_query, proj_key, proj_value



class AttentionLayer_pre(nn.Module):
    def __init__(self, embedding_dim, drop_ratio=0):
        super(AttentionLayer_pre, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, 16),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        out = self.linear(x)
        weight = F.softmax(out.view(1, -1), dim=1)
        return weight


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

