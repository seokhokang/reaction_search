import numpy as np
import pickle as pkl
import dgl, torch
import torch.nn.functional as F

from torch import nn
from torch.optim import Adam, SGD

from util import euclidean_sim


class Trainer:

    def __init__(self, net, model_path, mol_dict, cuda, lr = 1e-4, weight_decay = 1e-8, tau = 100):
    
        self.net = net.to(cuda)
        self.model_path = model_path
        self.mol_dict = mol_dict
        
        self.cuda = cuda
        
        self.batch_size = 4096
        self.tau = tau #temperature scaling in the SimCLR loss

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = Adam(self.net.parameters(), lr = lr, weight_decay = weight_decay)


    def load(self, model_path):

        checkpoint = torch.load(model_path)
        
        self.net.load_state_dict(checkpoint['graph_net'])
        if checkpoint['optimizer'] is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        
    def train(self, train_loader, valid_loader, max_epochs = 200, patience = 20):

        val_log = np.zeros(max_epochs)
        best_val_loss = 1e5
        for epoch in range(max_epochs):
        
            self.net.train()
            trn_loss = []
            
            for batch_idx, batch_data in enumerate(train_loader):

                product = self.net(batch_data[1].to(self.cuda), 0)
                reactant = self.net(batch_data[2].to(self.cuda), 1)
                #reagent = self.net(batch_data[3].to(self.cuda), 2)
                product_pred = reactant# + reagent

                features = torch.cat([product, product_pred], dim=0)
                loss = self._info_nce_loss(features)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
                trn_loss.append(loss.detach().cpu().numpy())

            self.net.eval()    
            val_loss = []
            
            with torch.no_grad():
                for batch_idx, batch_data in enumerate(valid_loader):
    
                    product = self.net(batch_data[1].to(self.cuda), 0)
                    reactant = self.net(batch_data[2].to(self.cuda), 1)
                    #reagent = self.net(batch_data[3].to(self.cuda), 2)
                    product_pred = reactant# + reagent

                    features = torch.cat([product, product_pred], dim=0)
                    loss = self._info_nce_loss(features)
                
                    val_loss.append(loss.cpu().numpy()) 

            val_log[epoch] = np.mean(val_loss)

            if val_log[epoch] < best_val_loss:
                best_val_loss = val_log[epoch]
                no_improve_cnt = 0

                checkpoint = { 
                    'epoch': epoch,
                    'graph_net': self.net.state_dict(),
                    'optimizer': self.optimizer.state_dict()
                }
                torch.save(checkpoint, self.model_path)
            
            else:
                no_improve_cnt += 1
            
            print('-- training epoch %d: trn_loss %.3f val_loss %.3f'%(epoch+1, np.mean(trn_loss), np.mean(val_loss)), '/', no_improve_cnt)
            
            if no_improve_cnt == patience:
                print('-- earlystopping')
                break

        self.load(self.model_path)  
  

    def update(self, qvec_list, rvec_list, rating_list, train_loader, max_iter = 100):

        print('Model Update -- NO. QUERIES: %d'%len(qvec_list))

        self.net.train()

        for m in self.net.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                #print('bn switch off') # to be removed
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False

        update_optimizer = SGD(self.net.parameters(), lr = 1e-4, weight_decay = 1e-8, momentum = 0.9)

        iter_cnt = 0
        while(iter_cnt < max_iter):

            for batch_idx, batch_data in enumerate(train_loader):

                # training minibatch
                product = self.net(batch_data[1].to(self.cuda), 0)
                reactant = self.net(batch_data[2].to(self.cuda), 1)
                #reagent = self.net(batch_data[3].to(self.cuda), 2)
                product_pred = reactant# + reagent
    
                features = torch.cat([product, product_pred], dim=0)
                loss_cl = self._info_nce_loss(features)
     
                # margin ranking loss
                loss_update = 0
                for j, (qvec, rvec, rating) in enumerate(zip(qvec_list, rvec_list, rating_list)):
                
                    # query eval data
                    q_product = self.net(qvec[1].to(self.cuda), 0)
                    q_reactant = self.net(qvec[2].to(self.cuda), 1)
                    #q_reagent = self.net(qvec[3].to(self.cuda), 2)
                    q_product_pred = q_reactant# + q_reagent
                    
                    has_product = q_product.abs().sum() > 0
                    has_reactant = q_reactant.abs().sum() > 0
                    if not has_reactant: q_product_pred = q_product
                    if not has_product: q_product = q_product_pred
                    q_embed = torch.cat([q_product, q_product_pred], 1)
                    
                    r_product = self.net(rvec[1].to(self.cuda), 0)
                    r_reactant = self.net(rvec[2].to(self.cuda), 1)
                    #r_reagent = self.net(rvec[3].to(self.cuda), 2)
                    r_product_pred = r_reactant# + r_reagent
                    
                    r_embed = torch.cat([r_product, r_product_pred], 1)
        
                    distmat = - euclidean_sim(q_embed, r_embed)
                    rating = rating.reshape(1, -1).to(self.cuda)

                    reldist = (rating - rating.T) * (distmat - distmat.T) + torch.abs(rating - rating.T) * 100 #margin
                    mask = torch.triu(torch.ones(reldist.shape, dtype=torch.bool), diagonal=1)

                    loss_update += torch.mean(torch.clamp(reldist[mask], min = 0, max = None))
                    
                loss_update = loss_update / len(qvec_list)

                # final loss
                loss = loss_cl + 0.01 * loss_update
    
                update_optimizer.zero_grad()
                loss.backward()
                update_optimizer.step()
                
                iter_cnt += 1
    
                if iter_cnt % 10 == 0:
                    print('-- updating iter %d: loss_cl %.3f, loss_update %.3f'%
                          (iter_cnt, loss_cl.detach().cpu().numpy(), loss_update.detach().cpu().numpy()))
                    
                if iter_cnt == max_iter: break

        checkpoint = { 
            'epoch': 'updated',
            'graph_net': self.net.state_dict(),
            'optimizer': None
        }
        torch.save(checkpoint, self.model_path)


    def embed(self, g_p = None, g_r = None, to_numpy = True):

        product = None
        product_pred = None
    
        self.net.eval()
        with torch.no_grad():
            if g_p is not None: product = self.net(g_p.to(self.cuda), 0).cpu()
            if g_r is not None: product_pred = self.net(g_r.to(self.cuda), 1).cpu()
        
        if to_numpy:
            if g_p is not None: product = product.numpy()
            if g_r is not None: product_pred = product_pred.numpy()
            
        return product, product_pred


    def _info_nce_loss(self, features):

        simmat = euclidean_sim(features)
        
        labels = torch.cat([torch.arange(len(features) // 2) for _ in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).bool()
        
        mask = torch.eye(labels.shape[0], dtype=torch.bool)
        labels = labels[~mask].view(labels.shape[0], -1)
        simmat = simmat[~mask].view(simmat.shape[0], -1)
            
        positives = simmat[labels].view(labels.shape[0], -1)
        negatives = simmat[~labels].view(simmat.shape[0], -1)
        
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.cuda)
        
        loss = self.loss_fn(logits/self.tau, labels)
        
        return loss