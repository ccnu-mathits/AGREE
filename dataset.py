'''
Created on Aug 8, 2016
Processing datasets.

@author: Xiangnan He (xiangnanhe@gmail.com)

Modified  on Nov 10, 2017, by Lianhai Miao
'''

import scipy.sparse as sp
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from collections import defaultdict

class GDataset(object):

    def __init__(self, dataname, user_path, group_path, user_in_group_path, num_negatives):
        '''
        Constructor
        '''
        self.dataname = dataname
        self.num_negatives = num_negatives
        # user data
        self.user_trainMatrix = self.load_rating_file_as_matrix(user_path + "Train.txt")
        # self.user_testRatings = self.load_rating_file_as_list(user_path + "Test.txt")
        self.user_testRatings, self.user_testNegatives = self.load_negative_file(user_path + "Negative.txt")
        self.num_users, self.num_items = self.user_trainMatrix.shape
        # group data
        self.group_trainMatrix = self.load_rating_file_as_matrix(group_path + "Train.txt")
        # self.group_testRatings = self.load_rating_file_as_list(group_path + "Test.txt")
        self.group_testRatings, self.group_testNegatives = self.load_negative_file(group_path + "Negative.txt")

        # self.user_ingroups_dict, self.gro_members_dict = self.get_group_member_dict(user_in_group_path)
        self.gro_members_dict, _ = self.get_group_member_dict(user_in_group_path)


    # def load_rating_file_as_list(self, filename):
    #     ratingList = []
    #     with open(filename, "r") as f:
    #         line = f.readline()
    #         while line != None and line != "":
    #             arr = line.split(" ")
    #             user, item = int(arr[0]), int(arr[1])
    #             ratingList.append([user, item])
    #             line = f.readline()
    #     return ratingList

    def load_negative_file(self, filename):
        testRatingList, negativeList = [], []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                if line == "\n":
                    line = f.readline()
                    continue
                if self.dataname == 'MaFengWo' or self.dataname == 'CAMRa2011':
                    arr = line.split(" ")
                    user, item = eval(arr[0])[0], eval(arr[0])[1]
                    testRatingList.append([user, item])
                    negatives = []
                    for x in arr[1:]:
                        negatives.append(int(x))
                    negativeList.append(negatives)
                elif self.dataname == 'ml-latest-small':
                    arr = line.split(' ')
                    user = int(arr[0])
                    pos_item = int(arr[1])
                    testRatingList.append([user, pos_item])
                    negatives = []
                    for x in arr[2:]:
                        negatives.append(int(x))
                    negativeList.append(negatives)
                line = f.readline()
        return testRatingList, negativeList

    def get_group_member_dict(self, user_in_group_path):
        g_m_d = defaultdict(list)
        u_g_d = defaultdict(list)
        with open(user_in_group_path, 'r') as f:
            line = f.readline().strip()
            while line != None and line != "":
                if self.dataname == 'MaFengWo' or self.dataname == 'CAMRa2011':
                    a = line.split(' ')
                    g = int(a[0])
                    # g_m_d[g] = []
                    for m in a[1].split(','):
                        g_m_d[g].append(int(m))
                        u_g_d[int(m)].append(g)
                elif self.dataname == 'ml-latest-small':
                    a = line.split(' ')
                    g = int(a[0])
                    # g_m_d[g] = []
                    for m in a[1:]:
                        g_m_d[g].append(int(m))
                        u_g_d[int(m)].append(g)
                line = f.readline().strip()
        return g_m_d, u_g_d


    def get_ui_dict(self, filename):
        # user_i -> items: 1: [10, 23]
        with open(filename, 'r') as reader:
            u_i_dict = defaultdict(list)
            i_u_dict = defaultdict(list)
            for line in reader:
                # 162540,32,4,...
                user_id, item_id = map(int, line.split(' ')[:2]) # ',' or '\t'
                u_i_dict[user_id].append(item_id)
                i_u_dict[item_id].append(user_id)

        return u_i_dict, i_u_dict

    # def get_group_member_dict(self, groupMember_file):
    #     with open(groupMember_file, 'r') as reader:
    #         g_m_dict = defaultdict(list)
    #         m_g_dict = defaultdict(list)
    #         for line in reader:
    #             group_id, member_ids = map(int, line.split(' ')[0]), map(int, line.split(','))
    #             for u in member_ids:
    #                 g_m_dict[group_id].append(u)


          
    #     return g_m_d, m_g_dict


    def load_rating_file_as_matrix(self, filename):
        # Get number of users and items
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(" ")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u) 
                num_items = max(num_items, i) 
                line = f.readline()
        # Construct matrix 
        mat = sp.dok_matrix((num_users + 1, num_items + 1), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(" ")
                if len(arr) > 2:
                    user, item, rating = int(arr[0]), int(arr[1]), int(arr[2])
                    if (rating > 0):
                        mat[user, item] = 1.0
                else:
                    user, item = int(arr[0]), int(arr[1])
                    mat[user, item] = 1.0
                line = f.readline()
        return mat

    def get_train_instances(self, train):
        user_input, pos_item_input, neg_item_input = [], [], []
        num_users = train.shape[0]
        num_items = train.shape[1]
        for (u, i) in train.keys():
            # positive instance
            for _ in range(self.num_negatives):
                pos_item_input.append(i)
            # negative instances
            for _ in range(self.num_negatives):
                j = np.random.randint(num_items)
                while (u, j) in train:
                    j = np.random.randint(num_items)
                user_input.append(u)
                neg_item_input.append(j)
        pi_ni = [[pi, ni] for pi, ni in zip(pos_item_input, neg_item_input)]
        return user_input, pi_ni

    def get_grouptrain_instances(self, train, num_items):
        user_input, pos_item_input, neg_item_input = [], [], []
        num_users = train.shape[0]
        for (u, i) in train.keys():
            # positive instance
            for _ in range(self.num_negatives):
                pos_item_input.append(i)
            # negative instances
            for _ in range(self.num_negatives):
                j = np.random.randint(num_items)
                while (u, j) in train:
                    j = np.random.randint(num_items)
                user_input.append(u)
                neg_item_input.append(j)
        pi_ni = [[pi, ni] for pi, ni in zip(pos_item_input, neg_item_input)]
        return user_input, pi_ni

    def get_user_dataloader(self, batch_size):
        user, positem_negitem_at_u = self.get_train_instances(self.user_trainMatrix)
        
        train_data = TensorDataset(torch.LongTensor(user), torch.LongTensor(positem_negitem_at_u))
        user_train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        return user_train_loader

    def get_group_dataloader(self, batch_size):
        # group and positem_negitem_at_g are two lists
        group, positem_negitem_at_g = self.get_train_instances(self.group_trainMatrix)
        train_data = TensorDataset(torch.LongTensor(group), torch.LongTensor(positem_negitem_at_g))
        group_train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
        
        return group_train_loader






