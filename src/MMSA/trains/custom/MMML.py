import logging

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from ...utils import MetricsTop, dict_to_str


class MMML():
    def __init__(self, args):
        self.args = args
        self.criterion = nn.L1Loss() if args.train_mode == 'regression' else nn.CrossEntropyLoss()
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)
        self.tasks = args.tasks
        
    def do_train(self, model, data_loader):    
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.args.learning_rate)

        total_loss = 0
        # Loop over all batches.         
        for batch in tqdm(data_loader):                    
            text_inputs = batch["text_tokens"].to(self.args.device)
            text_mask = batch["text_masks"].to(self.args.device)
            text_context_inputs = batch["text_context_tokens"].to(self.args.device)
            text_context_mask = batch["text_context_masks"].to(self.args.device)

            audio_inputs = batch["audio_inputs"].to(self.args.device)
            audio_mask = batch["audio_masks"].to(self.args.device)
            audio_context_inputs = batch["audio_context_inputs"].to(self.args.device)
            audio_context_mask = batch["audio_context_masks"].to(self.args.device)

            targets = batch["targets"].to(self.args.device).view(-1, 1)

            optimizer.zero_grad()                    # To zero out the gradients.

            if self.args.context:
                outputs = model(
                    text_inputs, text_mask, text_context_inputs, 
                    text_context_mask, audio_inputs, audio_mask, 
                    audio_context_inputs, audio_context_mask
                )
            else:
                outputs = model(
                    text_inputs, text_mask, audio_inputs, audio_mask
                )
            
            # Compute the training loss.
            if self.args.multi_task:
                loss = 0.0         
                for m in self.tasks:
                    sub_loss = self.args.loss_weights[m] * self.criterion(outputs[m], targets)
                    loss += sub_loss
    #                 train_loss[m] += sub_loss.item()*text_inputs.size(0)
                total_loss += loss.item()*text_inputs.size(0)  
            else:
                loss = self.criterion(outputs['M'], targets)        
                total_loss += loss.item()*text_inputs.size(0)
        
            loss.backward()                   
            optimizer.step()                
                
        total_loss = round(total_loss / len(data_loader.dataset), 4)
#         print('TRAIN'+" >> loss: ",total_loss)
        return total_loss

    def do_test(self, model, data_loader, mode):
        model.eval()   # Put the model in eval mode.
        if self.args.multi_task:
            y_pred = {'M': [], 'T': [], 'A': []}
            y_true = {'M': [], 'T': [], 'A': []}
            total_loss = 0
            val_loss = {
                'M':0,
                'T':0,
                'A':0
            }
        else:
            y_pred = []
            y_true = []
            total_loss = 0

        with torch.no_grad():
            for batch in tqdm(data_loader):                    # Loop over all batches.
                text_inputs = batch["text_tokens"].to(self.args.device)
                text_mask = batch["text_masks"].to(self.args.device)
                text_context_inputs = batch["text_context_tokens"].to(self.args.device)
                text_context_mask = batch["text_context_masks"].to(self.args.device)

                audio_inputs = batch["audio_inputs"].to(self.args.device)
                audio_mask = batch["audio_masks"].to(self.args.device)
                audio_context_inputs = batch["audio_context_inputs"].to(self.args.device)
                audio_context_mask = batch["audio_context_masks"].to(self.args.device)

                targets = batch["targets"].to(self.args.device).view(-1, 1)

                if self.args.context:
                    outputs = model(
                        text_inputs, text_mask, text_context_inputs, 
                        text_context_mask, audio_inputs, audio_mask, 
                        audio_context_inputs, audio_context_mask
                    )
                else:
                    outputs = model(
                        text_inputs, text_mask, audio_inputs, audio_mask
                    )
                
                # Compute loss.
                if self.args.multi_task:
                    loss = 0.0         
                    for m in self.tasks:
                        sub_loss = self.args.loss_weights[m] * self.criterion(outputs[m], targets)
                        loss += sub_loss
                        val_loss[m] += sub_loss.item()*text_inputs.size(0)
                    total_loss += loss.item()*text_inputs.size(0)
                    # add predictions
                    for m in self.tasks:
                        y_pred[m].append(outputs[m].cpu())
                        y_true[m].append(targets.cpu())
                else:
                    loss = self.criterion(outputs['M'], targets)        
                    total_loss += loss.item()*text_inputs.size(0)

                    # add predictions
                    y_pred.append(outputs['M'].cpu())
                    y_true.append(targets.cpu())

        if self.args.multi_task:
            for m in self.tasks:
                val_loss[m] = round(val_loss[m] / len(data_loader.dataset), 4)
            total_loss = round(total_loss / len(data_loader.dataset), 4)
            print(mode+" >> loss: ",total_loss, "   M_loss: ", val_loss['M'], "  T_loss: ", val_loss['T'], "  A_loss: ", val_loss['A'])

            eval_results = {}
            for m in self.tasks:
                pred, true = torch.cat(y_pred[m]), torch.cat(y_true[m])
                results = self.metrics(pred, true)
                print('%s: >> ' %(m) + dict_to_str(results))
                eval_results[m] = results
            eval_results = eval_results[self.tasks[0]]
            eval_results['Loss'] = total_loss 
        else:
            total_loss = round(total_loss / len(data_loader.dataset), 4)
            print(mode+" >> loss: ",total_loss)

            pred, true = torch.cat(y_pred), torch.cat(y_true)
            eval_results = self.metrics(pred, true)
            print('%s: >> ' %('M') + dict_to_str(eval_results))
            eval_results['Loss'] = total_loss
        
        return eval_results