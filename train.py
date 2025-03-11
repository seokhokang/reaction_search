"""A Trainer class that handles model training and fine-tuning operations, specialized in learning representations for chemical reactions.

Key Features:
- Train neural network models with contrastive loss for learning embeddings.
- Fine-tune models with margin ranking loss using user interactions.
- Save and load model checkpoints.
- Generate embeddings for chemical reaction components.

Class:
- Trainer: Training and fine-tuning a representation model.
"""

import numpy as np
import pickle as pkl
import dgl, torch
import torch.nn.functional as F

from torch import nn
from torch.optim import Adam, SGD

from util import euclidean_sim


class Trainer:
    """Trainer class for training and fine-tuning a representation model.
    
    This class provides functionalities for training and fine-tuning a representation model
    with contrastive loss (InfoNCE) and ranking-based margin loss.
    The model learns embeddings for chemical reactions.
    """

    def __init__(self, net, save_path, mol_dict, cuda, tau = 100):
        """Initializes an instance of the Trainer class.

        Args:
            net (torch.nn.Module): The neural network model.
            save_path (str): Path to save the trained model.
            mol_dict (dict): Dictionary mapping molecular indices to their SMILES representations.
            cuda (torch.device): Device to run the model (GPU/CPU).
            tau (float, optional): Temperature scaling hyperparameter for contrastive loss. Default is 100.
        """

        self.save_path = save_path
        self.mol_dict = mol_dict
        self.cuda = cuda
        self.tau = tau
        
        if net is not None: self.net = net.to(self.cuda)


    def load(self, model_path):
        """Loads a trained model from the specified file path.

        Args:
            model_path (str): Path to the saved model file.
        """

        self.net = torch.load(model_path).to(self.cuda)

        
    def train(self, train_loader, valid_loader, lr = 1e-4, weight_decay = 1e-8, max_epochs = 200, patience = 20):
        """Trains the model using contrastive representation learning.

        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader for training data.
            valid_loader (torch.utils.data.DataLoader): DataLoader for validation data.
            lr (float, optional): Learning rate. Default is 1e-4.
            weight_decay (float, optional): Weight decay for regularization. Default is 1e-8.
            max_epochs (int, optional): Maximum number of training epochs. Default is 200.
            patience (int, optional): Number of epochs to wait before early stopping. Default is 20.
        """

        optimizer = Adam(self.net.parameters(), lr = lr, weight_decay = weight_decay)

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
                try:
                    loss = self._info_nce_loss(features)
                except:
                    continue

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
                optimizer.step()
            
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

            if val_log[epoch] < best_val_loss - 1e-4:
                best_val_loss = val_log[epoch]
                no_improve_cnt = 0

                torch.save(self.net, self.save_path)
            
            else:
                no_improve_cnt += 1
            
            print('-- training epoch %d: trn_loss %.3f val_loss %.3f'%(epoch+1, np.mean(trn_loss), np.mean(val_loss)), '/', no_improve_cnt)
            
            if no_improve_cnt == patience:
                print('-- earlystopping')
                break

        self.load(self.save_path)  
  

    def update(self, qvec_list, rvec_list, rating_list, train_loader, max_iter = 100, margin_delta = 100, w_lambda = 0.01):
        """Fine-tunes the model using margin ranking loss with user-rated records.

        Args:
            qvec_list (torch.Tensor): Query vector embeddings.
            rvec_list (torch.Tensor): Record vector embeddings.
            rating_list (torch.Tensor): Ratings for query-record pairs.
            train_loader (torch.utils.data.DataLoader): DataLoader for training data.
            max_iter (int, optional): Maximum iterations for updating. Default is 100.
            margin_delta (float, optional): Margin hyperparameter for margin ranking loss. Default is 100.
            w_lambda (float, optional): Weight assigned to margin ranking loss. Default is 0.01.
        """

        print('Model Update -- NO. QUERIES: %d'%len(qvec_list))

        self.net.train()

        # switch off batch normalization
        for m in self.net.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
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

                    reldist = (rating - rating.T) * (distmat - distmat.T) + torch.abs(rating - rating.T) * margin_delta #margin
                    mask = torch.triu(torch.ones(reldist.shape, dtype=torch.bool), diagonal=1)

                    loss_update += torch.mean(torch.clamp(reldist[mask], min = 0, max = None))
                    
                loss_update = loss_update / len(qvec_list)

                # final loss
                loss = loss_cl + w_lambda * loss_update
    
                update_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
                update_optimizer.step()
                
                iter_cnt += 1
    
                if iter_cnt % 10 == 0:
                    print('-- updating iter %d: loss_cl %.3f, loss_update %.3f'%
                          (iter_cnt, loss_cl.detach().cpu().numpy(), loss_update.detach().cpu().numpy()))
                    
                if iter_cnt == max_iter: break

        torch.save(self.net, self.save_path)


    def embed(self, g_p = None, g_r = None, to_numpy = True):
        """Computes embeddings for given reactions.

        Args:
            g_p (torch.Tensor, optional): Graph representation of product molecules.
            g_r (torch.Tensor, optional): Graph representation of reactant molecules.
            to_numpy (bool, optional): If True, converts embeddings to NumPy array. Default is True.

        Returns:
            tuple: Product target and prediction vectors
        """

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
        """Computes the InfoNCE loss for contrastive learning using Euclidean similarity.

        Args:
            features (torch.Tensor): Feature representations of reactions.

        Returns:
            torch.Tensor: Computed InfoNCE loss.
        """

        loss_fn = torch.nn.CrossEntropyLoss()

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
        
        loss = loss_fn(logits/self.tau, labels)
        
        return loss