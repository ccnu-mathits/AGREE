'''
Created on Nov 10, 2017
Deal something

@author: Lianhai Miao
'''
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import math
import heapq
from collections import defaultdict

class AGREELoss(nn.Module):
    def __init__(self):
        super(AGREELoss, self).__init__()
    
    def forward(self, pos_preds, neg_preds):
        # https://github.com/LianHaiMiao/Attentive-Group-Recommendation/blob/master/main.py
        loss = torch.mean((pos_preds - neg_preds - 1).clone().pow(2))

        return loss

class BPRLoss(nn.Module):
    def __init__(self):
        super(BPRLoss, self).__init__()
    
    def forward(self, pos_preds, neg_preds):
        # https://github.com/guoyang9/BPR-pytorch/blob/master/main.py
        # loss = - (pos_preds - neg_preds).sigmoid().log().sum().clone()
        loss = - (pos_preds - neg_preds + 1e-8).sigmoid().log().mean().clone()
        # loss = - (pos_preds - neg_preds).log().mean().clone()
        return loss

class Helper(object):
    """
        utils class: it can provide any function that we need
    """
    def __init__(self):
        self.timber = True

    # The following functions are used to evaluate NCF_trans and group recommendation performance
    def evaluate_model(self, model, testRatings, testNegatives, K, type_m, device):
        """
        Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
        Return: score of each test rating.
        """
        hits, ndcgs = [], []

        for idx in range(len(testRatings)):
            (hr_list,ndcg_list) = self.eval_one_rating(model, testRatings, testNegatives, K, type_m, idx, device)

            hits.append(hr_list)
            ndcgs.append(ndcg_list)

        return list(np.array(hits).mean(0)), list(np.array(ndcgs).mean(0))


    def eval_one_rating(self, model, testRatings, testNegatives, K, type_m, idx, device):
        p = 0
        rating = testRatings[idx]
        items = testNegatives[idx]
        u = rating[0]
        gtItem = rating[1]
        items.append(gtItem)
        # Get prediction scores
        map_item_score = {}
        users = np.full(len(items), u)

        users_var = torch.from_numpy(users).long().to(device)
        items_var = torch.LongTensor(items).to(device)

        # get the predictions from the trained model
        if type_m == 'group':
            predictions = model(users_var, items_var, 'group')

        if type_m == 'group_fixed_agg':
            predictions = model(users_var, items_var, 'group_fixed_agg')   
            
        elif type_m == 'sa_group':
            predictions = model(users_var, items_var, 'sa_group')  
        elif type_m == 'H-fixed-agg-GR':
            predictions = model(users_var, items_var, 'H-fixed-agg-GR')
        elif type_m == 'target_user_HA_improved':
            predictions = model(users_var, items_var, 'target_user_HA_improved')
        elif type_m == 'target_user_HA':
            predictions = model(users_var, items_var, 'target_user_HA')    
        elif type_m == 'user':
            predictions = model(users_var, items_var, 'user')

        for i in range(len(items)):
            item = items[i]
            map_item_score[item] = predictions.cpu().data.numpy()[i]
        items.pop() # delete the last item in the list items
        
        hr_list, ndcg_list = [], []
        for topk in K:
            # Evaluate top rank list
            ranklist = heapq.nlargest(topk, map_item_score, key=map_item_score.get)
            hr = self.getHitRatio(ranklist, gtItem)
            ndcg = self.getNDCG(ranklist, gtItem)
            hr_list.append(hr)
            ndcg_list.append(ndcg)
        return (hr_list, ndcg_list)

    def getHitRatio(self, ranklist, gtItem):
        for item in ranklist:
            if item == gtItem:
                return 1
        return 0

    def getNDCG(self, ranklist, gtItem):
        for i in range(len(ranklist)):
            item = ranklist[i]
            if item == gtItem:
                return math.log(2) / math.log(i+2)
        return 0