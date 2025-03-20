import os
import torch
from data_process import *
import pandas as pd
from models.MiTCD_exp import cMLP,CAGKE_U,train_model_formal,DREC,pre_train_model_v2,CAGKE_Multi,train_model_formal_for_vis
from synthetic import simulate_var,simulate_lorenz_96
import matplotlib.pyplot as plt
from sklearn import preprocessing
import random
import numpy as np
import time
import argparse
from collections import OrderedDict

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

fix_seed = 230526
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='MiTCD')
# data loader
parser.add_argument('--length', type=int, default=1000,
                    help='overall_length of time series')
parser.add_argument('--length_per_batch', type=int, default=50,help='length of time series per batch')
parser.add_argument('--dataname', type=str, default='Lorenz',help='dataset name')
parser.add_argument('--dis_dim_pat', type=str, default='random',help='discretization mode for mixed time series generation')
parser.add_argument('--p', type=int, default=10, help = 'variable dimension')
parser.add_argument('--disp', type=int, default=4, help = 'dis variable dimensions')
parser.add_argument('--lag', type=int, default=3, help = 'VAR lag')
parser.add_argument('--F', type=float, default=10, help = 'F of lorenz')

# model
# cagke
parser.add_argument('--embed_dim', type = int, default = 10, help = 'embed dimension of CAGKE')
parser.add_argument('--sigma_min', type = float, default = 0.3,help = 'ini min bandwidth of CAGKE')
parser.add_argument('--sigma_max', type = float, default = 4.0, help = 'ini max bandwidth of CAGKE')
parser.add_argument('--noise_sigma', type = float, default = 0.01, help = 'noise magnitude of CAGKE')
parser.add_argument('--multi_cagke',type = bool, default = True, help = 'multi-cagke or unified cagke flag')
parser.add_argument('--cagke_learn_flag',type = bool, default = True, help = 'learnable cagke flag')

# predictors

parser.add_argument('--lag_cmlp', type = int, default = 5, help = 'max lag of predictor/decoder')


# pre-training
parser.add_argument('--pre_lr',type = float, default= 0.01, help = 'learning rate of pre-trainiong stage')
parser.add_argument('--pre_max_iter', type = int, default = 200, help = 'max iteration of pre-training stage')
parser.add_argument('--pre_rec_lambda',type = float, default= 0.5, help = 'reconstruction weight of pre-training stage')

# causal-training
parser.add_argument('--lr',type = float, default= 0.01, help = 'learning rate of causal discovery learning stage')
parser.add_argument('--lambda_sparse',type = float, default= 0.02, help = 'sparse constraint weight of pre-training stage')
parser.add_argument('--lambda_ridge',type = float, default= 0.01, help = 'ridge constraint weight of pre-training stage')
parser.add_argument('--cagke_up_period', type = int, default = 1000, help = 'update period for cagke')
parser.add_argument('--max_iter', type = int, default = 50000, help = 'max iteration of causal discovery learning stage')
parser.add_argument('--rec_lambda',type = float, default= 0.1, help = 'reconstruction weight of causal discovery learning stage')
parser.add_argument('--penalty', type = str, default = 'H', help = 'sparse penalty type')
parser.add_argument('--check_every', type = int, default = 100, help = 'number of epochs for checking results')

parser.add_argument('--device', type = str, default = 'cuda:3')

# others
parser.add_argument('--r',type = float, default= 0.8)
parser.add_argument('--lr_min',type = float, default= 1e-8)
parser.add_argument('--sigma',type = float, default= 0.5)
parser.add_argument('--lr_decay',type = float, default= 0.5)
parser.add_argument('--switch_tol',type = float, default= 1e-3)
parser.add_argument('--m', type = int, default = 10)
parser.add_argument('--verbose', type = int, default = 1)
parser.add_argument('--begin_line_search', type=bool, default = True)
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')


args = parser.parse_args()


# generate typeflag to denote index of variable type
typeflag = []

for i in range(args.disp):
    typeflag.append(0)
for i in range(args.disp,args.p):
    typeflag.append(1)

if args.dis_dim_pat == 'random':
    random.shuffle(typeflag)

args.typeflag = typeflag

#  generate data
if args.dataname == 'VAR':

    X_real, X, GC = generate_mixed_var(p=args.p,typeflag=args.typeflag,T=args.length,
                                   lag=args.lag,
                                   length_per_batch=args.length_per_batch,device =args.device)


if args.dataname == 'Lorenz':

    X_real, X, GC = generate_mixed_lorenz_96(p=args.p,typeflag=args.typeflag,T=args.length,
                                   F = args.F,
                                   length_per_batch=args.length_per_batch,device =args.device)

if args.dataname == 'fMRI':
    X_real, X, GC = generate_mixed_fMRI(p=args.p, typeflag=args.typeflag, length_per_batch=args.length_per_batch,
                       device=args.device)

if args.dataname == 'LV':
    X_real, X, GC = generate_mixed_LotkaVolterra(p=args.p, typeflag=args.typeflag, T=args.length, length_per_batch=args.length_per_batch,
                       device=args.device)

if args.dataname == 'DREAM3E1':
    X_real, X, GC = generate_mixed_dream3e1(p=args.p, typeflag=args.typeflag, length_per_batch=args.length_per_batch,
                       device=args.device)


if args.dataname == 'DREAM3E2':
    X_real, X, GC = generate_mixed_dream3e2(p=args.p, typeflag=args.typeflag,  length_per_batch=args.length_per_batch,
                       device=args.device)

if args.dataname == 'DREAM3Y1':
    X_real, X, GC = generate_mixed_dream3y1(p=args.p, typeflag=args.typeflag,  length_per_batch=args.length_per_batch,
                       device=args.device)

if args.dataname == 'DREAM3Y2':
    X_real, X, GC = generate_mixed_dream3y2(p=args.p, typeflag=args.typeflag,  length_per_batch=args.length_per_batch,
                       device=args.device)

if args.dataname == 'DREAM3Y3':
    X_real, X, GC = generate_mixed_dream3y3(p=args.p, typeflag=args.typeflag,  length_per_batch=args.length_per_batch,
                       device=args.device)



args.typeflag = typeflag
args.hidden = args.p*2

print('Args in experiment:')
print(args)
setting = 'sigma_min{}_lag_cmlp{}_hidden{}_lr_{}lambda_sparse_{}'.format(
    args.sigma_min,
    args.lag_cmlp,
    args.hidden,
    args.lr,
    args.lambda_sparse)

num_dis_series = args.disp

print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))

cagke = CAGKE_Multi(length=args.length_per_batch, num_dis=num_dis_series,
                    embed_dim=args.embed_dim, sigma_min=args.sigma_min,
                    sigma_max=args.sigma_max,
                    noise_sigma=args.noise_sigma).cuda(device=args.device)

cmlp = cMLP(args.p, args.lag_cmlp, [args.hidden],
            embed_dim=args.embed_dim,
            length=args.length_per_batch).cuda(device=args.device)

drec = DREC(num_dis_series, args.lag_cmlp, [args.hidden],
            embed_dim=args.embed_dim,
            length=args.length_per_batch).cuda(device=args.device)

train_loss_list, forecast_loss_list, recons_loss_list, recon_acc_list, cagke_sigma_list, cagke_weight_list = \
    pre_train_model_v2(cagke, cmlp, drec, X, lr=args.pre_lr, type_flag=args.typeflag,
                        multi_cagke=args.multi_cagke,
                       max_iter=args.pre_max_iter, rec_lambda=args.pre_rec_lambda,
                       check_every=1)


cmlp = cMLP(args.p, args.lag_cmlp, [args.hidden],
            embed_dim=args.embed_dim,
            length=args.length_per_batch).cuda(device=args.device)

causal_train_loss_list, causal_train_mse_list, auroc_best, auprc_best, auroc_list, auprc_list, non_smooth_list = \
    train_model_formal_for_vis(cagke, cmlp, drec, X, type_flag=args.typeflag, lam=args.lambda_sparse,
                       lam_ridge=args.lambda_ridge,
                       lr=args.lr, penalty=args.penalty,
                       max_iter = args.max_iter, GC_True=GC, multi_cagke=args.multi_cagke, cagke_learn_flag=args.cagke_learn_flag, cagke_up_period=args.cagke_up_period,
                       rec_lambda=args.rec_lambda,
                       check_every=10, r=args.r, lr_min=args.lr_min,
                       sigma=args.sigma,
                       monotone=False, m=args.m,
                       lr_decay=args.lr_decay,
                       begin_line_search=args.begin_line_search,
                       switch_tol=args.switch_tol, verbose=args.verbose)

res = {
    't_now': time.asctime(),
    'data':args.dataname,
    'p':args.p,
    'dis_p':args.disp,
    'T':args.length,
    'typeflag':args.typeflag,
    'sigma_min': args.sigma_min,
    'lag_cmlp': args.lag_cmlp,
    'hidden': args.hidden,
    'pre_lr': args.pre_lr,
    'lr': args.lr,
    'lambda_sparse': args.lambda_sparse,
    'best_auroc':auroc_best,
    'best_auprc':auprc_best
}


expd_path = './result_mitcd/' + str(args.dataname)+'p='+str(args.p)+'dis_p='+str(args.disp)+'T='+str(args.length)+'F='+str(args.F)
if not os.path.exists(expd_path):
    os.makedirs(expd_path)

expd_pathh = expd_path + '/result.txt'

with open(expd_pathh, 'a') as f:
    f.write('\n')
    for k, v in res.items():
        f.write(k + ':' + str(v) + '   ')
    f.write('\n')

save_root = './result_mitcd/' + str(args.dataname)+'p='+str(args.p)+'dis_p='+str(args.disp)+'T='+str(args.length)+'F='+str(args.F) + '/trained_models'
if not os.path.exists(save_root):
    os.makedirs(save_root)
save_roott = save_root + '/AUROC=' + str(res['best_auroc']) + '.pth'

torch.save(
    {'cmlp': cmlp, 'cagke': cagke, 'dic_rec': drec, 'GC_true': GC, 'type_flag': args.typeflag,
     'pre_f_loss_list': forecast_loss_list, 'pre_rec_loss_list': recons_loss_list,
     'pre_rec_acc_list': recon_acc_list,
     'pre_sigma_list': cagke_sigma_list, 'pre_weight_list': cagke_weight_list,
     'causal_train_loss_list': causal_train_loss_list, 'causal_train_mse_list': causal_train_mse_list,
     'auroc_list': auroc_list, 'auprc_list': auprc_list, 'non_smooth_list': non_smooth_list}, save_roott)
