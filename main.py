import faulthandler; faulthandler.enable()
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from time import time
import os
from collections import Counter
# import matplotlib.pyplot as plt
import pandas as pd
from model.agree import AGREE
from model.gradi import GRADI
from utils.util import Helper, AGREELoss, BPRLoss
from dataset import GDataset
import argparse
# dataname = 'MaFengWo'
# dataname = 'ml-latest-small'
dataname = 'CAMRa2011'
parser = argparse.ArgumentParser()
parser.add_argument('--save_name', type=str, default='AGREE-mf')
parser.add_argument('--path', type=str, default='/home/admin123/ruxia/HAN-CDGRccnu/Experiments/AGREE/data/' + dataname)
parser.add_argument('--user_dataset', type=str, default= '/home/admin123/ruxia/HAN-CDGRccnu/Experiments/AGREE/data/' + dataname + '/userRating')
parser.add_argument('--group_dataset', type=str, default= '/home/admin123/ruxia/HAN-CDGRccnu/Experiments/AGREE/data/' + dataname + '/groupRating')
parser.add_argument('--user_in_group_path', type=str, default= '/home/admin123/ruxia/HAN-CDGRccnu/Experiments/AGREE/data/' + dataname + '/groupMember.txt')

parser.add_argument('--embedding_size_list', type=list, default=[32])
parser.add_argument('--n_epoch', type=int, default=800)
parser.add_argument('--early_stop', type=int, default=20)
parser.add_argument('--num_negatives', type=list, default=4)
parser.add_argument('--batch_size', type=list, default=128)
parser.add_argument('--lr', type=list, default=[0.002])
# parser.add_argument('--lr', type=list, default=[0.000005]) # CAMRa2011 dataset learning rate
# parser.add_argument('--lr', type=list, default=[0.00005, 0.000005, 0.0000005])
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--drop_ratio_list', type=list, default=[0.2])
parser.add_argument('--topK_list', type=list, default=[5, 10])
parser.add_argument('--lmd_list', type=list, default=[1.0]) # group aggregator
parser.add_argument('--eta_list', type=list, default=[1])

# parser.add_argument('--type_m_gro', type=str, default='group')
# parser.add_argument('--type_m_usr', type=str, default='user')

parser.add_argument('--type_m_gro', type=str, default='user')
parser.add_argument('--type_m_usr', type=str, default='user')

args = parser.parse_args()

DEVICE = torch.device('cuda:{}'.format(args.cuda)) if torch.cuda.is_available() else torch.device('cpu')

# DEVICE = torch.device('cpu')

# train the model
def training(model, train_loader, epoch_id, type_m, lr):

    # optimizer = optim.RMSprop(model.parameters(), lr)
    optimizer = optim.Adam(model.parameters(), lr)

    # loss function
    loss_function = AGREELoss()
    # loss_function = BPRLoss()

    losses = []   
    for batch_id, (u, pi_ni) in enumerate(train_loader): 
        # Data Load
        user_input = u
        pos_item_input = pi_ni[:, 0]
        neg_item_input = pi_ni[:, 1]

        user_input, pos_item_input, neg_item_input = user_input.to(DEVICE), pos_item_input.to(DEVICE), neg_item_input.to(DEVICE)
        # Forward
        
        pos_prediction = model(user_input, pos_item_input, type_m)
        neg_prediction = model(user_input, neg_item_input, type_m)
        
        # Zero_grad
        model.zero_grad()
        # Loss value of one batch of examples
        loss = loss_function(pos_prediction, neg_prediction)
        # record loss history
        losses.append(loss)  
        # Backward
        loss.backward(torch.ones_like(loss))
        # Update parameters
        optimizer.step()

    print('Iteration %d, loss is [%.4f ]' % (epoch_id, torch.mean(torch.tensor(losses))))



def evaluation(model, helper, testRatings, testNegatives, K, type_m, DEVICE):
    model = model.to(DEVICE)
    # set the module in evaluation mode
    model.eval()
    (hrs, ndcgs) = helper.evaluate_model(model, testRatings, testNegatives, K, type_m, DEVICE)
    # hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    return hrs, ndcgs


if __name__ == '__main__':
    helper = Helper()
    # initial dataSet class
    dataset = GDataset(dataname, args.user_dataset, args.group_dataset, args.user_in_group_path, args.num_negatives)
    g_m_d = dataset.gro_members_dict
    num_groups = len(g_m_d)
    num_users, num_items = dataset.num_users, dataset.num_items
    print('Data prepare is over!')

    
    
    dir_name = os.path.join(args.path, args.save_name)

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    
    for lr in args.lr:
        for drop_ratio in args.drop_ratio_list:
            for lmd in args.lmd_list:
                for eta in args.eta_list:
                    for embedding_size in args.embedding_size_list:
                        for i in range(5):
                            # build AGREE model
                            model = AGREE(num_users, num_items, num_groups, embedding_size, g_m_d, DEVICE, drop_ratio, lmd, eta).to(DEVICE)
                            # model = GRADI(num_users, num_items, num_groups, embedding_size, g_m_d, DEVICE, drop_ratio, lmd, eta).to(DEVICE)
                            # args information
                            print("Model at embedding size %d, run Iteration:%d, drop_ratio at %1.2f, lmd at %1.2f, eta at %1.2f, lr at %1.6f" %(embedding_size, args.n_epoch, drop_ratio, lmd, eta, lr))

                            # train the model
                            HR_gro = []
                            NDCG_gro = []
                            HR_user = []
                            NDCG_user = []
                            user_train_time = []
                            gro_train_time = []
                            best_hr_gro = 0
                            best_ndcg_gro = 0
                            stop = 0
                            for epoch in range(args.n_epoch):
                                # set the module in training mode
                                model.train()
                                # 开始训练时间
                                t1_user = time()
                                # train the user
                                training(model, dataset.get_user_dataloader(args.batch_size), epoch, args.type_m_usr, lr)
                                # print("user training time is: [%.1f s]" % (time()-t1_user))
                                user_train_time.append(time()-t1_user)

                                t1_gro = time()
                                # train the group
                                training(model, dataset.get_group_dataloader(args.batch_size), epoch, args.type_m_gro, lr)
                                # print("group training time is: [%.1f s]" % (time()-t1_gro))
                                gro_train_time.append(time()-t1_gro)

                                # evaluation
                                t2 = time()
                                u_hr, u_ndcg = evaluation(model, helper, dataset.user_testRatings, dataset.user_testNegatives, args.topK_list, args.type_m_usr, DEVICE)
                                HR_user.append(u_hr)
                                NDCG_user.append(u_ndcg)                                
                                # print('User Iteration %d [%.1f s]: HR_user NDCG_user' % (
                                #     epoch, time() - t1_user))
                                # print(HR_user[-1], NDCG_user[-1])

                                t3 = time()
                                hr, ndcg = evaluation(model, helper, dataset.group_testRatings, dataset.group_testNegatives, args.topK_list, args.type_m_gro, DEVICE)
                                HR_gro.append(list(hr))
                                NDCG_gro.append(list(ndcg))
                                # print('Group Iteration %d [%.1f s]: HR_group NDCG_group' % (epoch, time() - t1_user))
                                # print(HR_gro[-1], NDCG_gro[-1])

                                if hr[0] > best_hr_gro:
                                    best_hr_gro = hr[0]
                                    best_ndcg_gro = ndcg[0]
                                    stop = 0
                                else:
                                    stop = stop + 1                                            
                                print('Test HR_user:', u_hr, '| Test NDCG_user:', u_ndcg)
                                print('Test HR_gro:', hr, '| Test NDCG_gro:', ndcg)
                                if stop >= args.early_stop:
                                    print('*' * 20, 'stop training', '*' * 20)
                                    print('Group Iteration %d [%.1f s]: HR_group NDCG_group' % (epoch, time() - t1_user))
                                    print('Best HR_gro:', HR_gro[-1], '| Best NDCG_gro:', NDCG_gro[-1])
                                    break

                            
                            # EVA_user = np.column_stack((HR_user, NDCG_user, user_train_time))
                            # EVA_gro = np.column_stack((HR_gro, NDCG_gro, gro_train_time))

                            EVA_data = np.column_stack((HR_user, NDCG_user, HR_gro, NDCG_gro))

                            print("save to file...")

                            filename = "Adam_%s_%s_E%d_batch%d_drop_ratio_%1.2f_lambda_%1.2f_eta_%1.2f_lr_%1.6f_%d" % (args.type_m_gro, args.type_m_usr, embedding_size, args.batch_size, drop_ratio, lmd, eta, lr, i)

                            filename = os.path.join(dir_name, filename)
                            
                            # np.savetxt(filename, EVA_user, fmt='%1.4f', delimiter=' ', header='hr5_user, hr10_user, ndcg5_user, ndcg10_user, utime, hr5_gro, hr10_gro, ndcg5_gro, ndcg10_gro, gtime')
                            np.savetxt(filename, EVA_data, fmt='%1.4f', delimiter=' ')

                            # np.savetxt('./EVA_user_2sa_mfw.txt', EVA_user, delimiter=' ', fmt="%.4f")
                            # np.savetxt('./EVA_gro_2sa_mfw.txt', EVA_gro, delimiter=' ', fmt="%.4f")

                            print("Done!")
