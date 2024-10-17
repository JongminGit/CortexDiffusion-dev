import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import compute_dice, coords_normalize
from utils import read_vtk, write_vtk

lossfunc = nn.CrossEntropyLoss()

def train(net, dataset, batch_size, optimizer, epoch, device):
    net.train()
    Loss = 0
    Acc = 0

    dataloader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True)
    lut = dict()
    lut.update({0:0,2:1,3:2,5:3,6:4,7:5,8:6,9:7,10:8,11:9,12:10,13:11,14:12,15:13,16:14,17:15,18:16,19:17,20:18,21:19,22:20,23:21,24:22,25:23,26:24,27:25,28:26,29:27,30:28,31:29,34:30,35:31})
    
    for _, coords, labels, mass, evals, evecs, gradX, gradY in tqdm(dataloader, desc=str(epoch)):
        labels = torch.from_numpy(np.array([lut[l] for l in labels.squeeze(0).numpy()])).unsqueeze(0)
        coords = coords.to(torch.device(device))
        labels = labels.to(torch.device(device))
        mass = mass.to(torch.device(device))
        evals = evals.to(torch.device(device))
        evecs = evecs.to(torch.device(device))
        gradX = gradX.to(torch.device(device))
        gradY = gradY.to(torch.device(device))

        optimizer.zero_grad()
        outputs = net(coords, mass, evals, evecs, gradX, gradY)  # (B,Nv,C_out)
        loss = lossfunc(outputs.transpose(2,1), labels)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(outputs, dim=2)
        acc = torch.sum((preds == labels).sum(dim=1) / labels.shape[1])
        Loss += loss.detach().item()
        Acc += acc.detach().item()
    
    Loss /= dataset.__len__()
    Acc /= dataset.__len__()

    print('train: ce_loss = {:.6f} | acc = {:.2f} %'.format(Loss, 100*Acc))

    return Loss, Acc

def val(net, dataset, batch_size, epoch, device):
    net.eval()
    Loss = 0
    Acc = 0
    Dice = 0

    dataloader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True)
    lut = dict()
    lut.update({0:0,2:1,3:2,5:3,6:4,7:5,8:6,9:7,10:8,11:9,12:10,13:11,14:12,15:13,16:14,17:15,18:16,19:17,20:18,21:19,22:20,23:21,24:22,25:23,26:24,27:25,28:26,29:27,30:28,31:29,34:30,35:31})

    with torch.no_grad():
        for _, coords, labels, mass, evals, evecs, gradX, gradY in tqdm(dataloader, desc=str(epoch)):
            labels = torch.from_numpy(np.array([lut[l] for l in labels.squeeze(0).numpy()])).unsqueeze(0)
            coords = coords.to(torch.device(device))
            labels = labels.to(torch.device(device))
            mass = mass.to(torch.device(device))
            evals = evals.to(torch.device(device))
            evecs = evecs.to(torch.device(device))
            gradX = gradX.to(torch.device(device))
            gradY = gradY.to(torch.device(device))

            outputs = net(coords, mass, evals, evecs, gradX, gradY)  # (B,Nv,C_out)
            preds = torch.argmax(outputs, dim=2)
            acc = torch.sum((preds == labels).sum(dim=1) / labels.shape[1])
            loss = lossfunc(outputs.transpose(2,1), labels)

            for b in range(batch_size):
                Dice +=  compute_dice(preds[b], labels[b], net.out_channels)

            Loss += loss.item()
            Acc += acc.item()
    
    Loss /= dataset.__len__()
    Acc /= dataset.__len__()
    Dice /= dataset.__len__()

    print('val: ce_loss = {:.6f} | acc = {:.2f} % | dice = {:.2f} %'.format(Loss, 100*Acc, 100*Dice))

    return Loss, Acc, Dice

def test(net, dataset, epoch, device, source_vtk_root, target_vtk_root):
    net.eval()
    #acc = []
    Dice = []

    dataloader = DataLoader(dataset=dataset,batch_size=1,shuffle=False)
    lut = dict()
    lut.update({0:0,2:1,3:2,5:3,6:4,7:5,8:6,9:7,10:8,11:9,12:10,13:11,14:12,15:13,16:14,17:15,18:16,19:17,20:18,21:19,22:20,23:21,24:22,25:23,26:24,27:25,28:26,29:27,30:28,31:29,34:30,35:31})

    with torch.no_grad():
        for basename, coords, labels, mass, evals, evecs, gradX, gradY in tqdm(dataloader, desc=str(epoch)):
            labels = torch.from_numpy(np.array([lut[l] for l in labels.squeeze(0).numpy()])).unsqueeze(0)
            coords = coords.to(torch.device(device))
            labels = labels.to(torch.device(device))
            mass = mass.to(torch.device(device))
            evals = evals.to(torch.device(device))
            evecs = evecs.to(torch.device(device))
            gradX = gradX.to(torch.device(device))
            gradY = gradY.to(torch.device(device))

            outputs = net(coords, mass, evals, evecs, gradX, gradY)  # (1,Nv,C_out)
            preds = torch.argmax(outputs, dim=2).squeeze(0)  # (Nv)

            source_vtk = read_vtk(os.path.join(source_vtk_root, basename[0] + '.vtk'))
            target_vtk = dict()
            target_vtk['vertices'] = coords_normalize(source_vtk['vertices'])
            target_vtk['faces'] = source_vtk['faces']
            target_vtk['label'] = preds.cpu().numpy()

            write_vtk(target_vtk, 'vertices', os.path.join(target_vtk_root, basename[0] + '.vtk'))
            #dice = compute_dice(preds[0], labels[0], net.out_channels)
            #Dice.append(dice)
            #with open(os.path.join(checkpoint, "subject_logs.txt"), 'a+') as f:
            #    print(basename[0] + ' : dice = {:.2f} %'.format(100*np.mean(dice)),file=f)
            
            #acc.append(torch.sum((preds == labels).sum(dim=1) / labels.shape[1]).item())
            #dice.append(compute_dice(preds[0], labels[0], net.out_channels))

    #return Dice
