import os
import numpy as np
import torch
import torch.optim as optim
from step import train, val
from dataset import Data
from model import DiffusionNet
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

batch_size = 1  # We suggest you set the batch size to 1, since the number of vertices in different cortical surface may not be the same
device = 'cuda:0'

checkpoint = r"/data/lfs/kimjongmin8/Develop/CortexDiffusion/logs_white2"
data_path = r"/data/lfs/kimjongmin8/Develop/CortexDiffusion/Mindboggle_dataset/white"
folds = os.listdir(data_path)  # len(folds) = 5 # Assuming your data has been divided into 5 folds in advance and placed in different folders under 'data_path' 
f = open(os.path.join(checkpoint,"logs.txt"), "w+")

for i in range(5):
    # We employed 5-fold cross-validation for performance quantification
    if i == 0:
        l = folds
    else:
        l = folds[i:] + folds[:i]
    test_fold = l[0]
    val_fold = l[1]
    train_fold = l[2:]
    
    train_loss = None
    val_loss = None
    val_dice = None
    val_acc = None
    Loss_train = []
    Loss_val = []

    train_dataset = Data(root=data_path, folds=train_fold, k=120)
    val_dataset = Data(root=data_path, folds=val_fold, k=120)
    test_dataset = Data(root=data_path, folds=test_fold, k=120)

    net = DiffusionNet(in_channels=3, out_channels=32, hidden_channels=128, n_block=4, dropout=True, with_gradient_features=True)
    net.to(torch.device(device))
    save_path_train = os.path.join(checkpoint, 'net_train_' + test_fold + '.pkl')
    save_path_val = os.path.join(checkpoint, 'net_val_' + test_fold + '.pkl')

    ########## train ##########
    print('now test fold is ' + test_fold)
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,mode="min",factor=0.5,patience=2)
    for i in range(100):
        print('now lr = ' + str(optimizer.state_dict()['param_groups'][0]['lr']))
        if optimizer.state_dict()['param_groups'][0]['lr'] < 1e-6:
            break
        loss_train, acc_train = train(net, train_dataset, batch_size, optimizer, i+1, device)
        loss_val, acc_val, dice_val = val(net, val_dataset, batch_size, i+1, device)
        scheduler.step(loss_val)
    
        Loss_train.append(loss_train)
        Loss_val.append(loss_val)
    
        if train_loss is None or loss_train < train_loss:
            train_loss = loss_train
            torch.save(net.state_dict(),save_path_train)
        
        if val_loss is None or dice_val >= val_dice:
            val_loss = loss_val
            val_dice = dice_val
            val_acc = acc_val
            torch.save(net.state_dict(),save_path_val)
    
    net.load_state_dict(torch.load(save_path_val, map_location=device))
    _, _, test_dice = val(net, test_dataset, batch_size, 1, device)
        
    with open(os.path.join(checkpoint,"logs.txt"), 'a+') as f:
        print('now test fold is ' + test_fold + ': train_acc = {:.2f} %'.format(100*acc_train) + ' | val_dice = {:.2f} %'.format(100*val_dice) + ' | test_dice = {:.2f} %'.format(100*test_dice),file=f)
    
    data = dict()
    data['loss_train'] = Loss_train
    data['loss_val'] = Loss_val
    np.save(os.path.join(checkpoint,"experiment_data_" + test_fold + ".dict"),data)