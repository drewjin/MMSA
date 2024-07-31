import traceback

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from datetime import datetime
import json
import os
import sys
import time
import h5py
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import trange
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from graph_model.iemocap_inverse_sample_count_ce_loss import IEMOCAPInverseSampleCountCELoss
from consts import GlobalConsts as gc
from model import NetMTGATAverageUnalignedConcatMHA
from dataset.MOSEI_dataset import MoseiDataset
from dataset.MOSEI_dataset_unaligned import MoseiDatasetUnaligned
from dataset.MOSI_dataset import MosiDataset
from dataset.MOSI_dataset_unaligned import MosiDatasetUnaligned
from dataset.IEMOCAP_dataset import IemocapDatasetUnaligned, IemocapDataset
import logging
import util
import pathlib

import standard_grid


def multiclass_acc(preds, truths):
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))


def weighted_acc(preds, truths):
    preds, truths = preds > 0, truths > 0
    tn, fp, fn, tp = confusion_matrix(truths, preds).ravel()
    n, p = len([i for i in preds if i == 0]), len([i for i in preds if i > 0])
    return (tp * n / p + tn) / (2 * n)

def eval_iemocap(split, output_all, label_all, epoch=None):
    truths = np.array(label_all)
    results = np.array(output_all)
    test_preds = results.reshape((-1, 4, 2))
    test_truth = truths.reshape((-1, 4))
    emos_f1 = {}
    emos_acc = {}
    for emo_ind, em in enumerate(gc.best.iemocap_emos):
        test_preds_i = np.argmax(test_preds[:, emo_ind], axis=1)
        test_truth_i = test_truth[:, emo_ind]
        # if epoch != None and epoch % 5 == 0 and split == 'test':
        #     # import ipdb
        #     # ipdb.set_trace()
        f1 = f1_score(test_truth_i, test_preds_i, average='weighted')
        emos_f1[em] = f1
        acc = accuracy_score(test_truth_i, test_preds_i)
        emos_acc[em] = acc
        logging.info("\t%s %s F1 Score: %f" % (split, gc.best.iemocap_emos[emo_ind], f1))
        logging.info("\t%s %s Accuracy: %f" % (split, gc.best.iemocap_emos[emo_ind], acc))
    return emos_f1, emos_acc


def eval_mosi_mosei(split, output_all, label_all):
    # The length of output_all / label_all is the number
    # of samples within that split
    truth = np.array(label_all)
    preds = np.array(output_all)
    mae = np.mean(np.abs(truth - preds))
    cor = np.corrcoef(preds, truth)[0][1]
    acc = accuracy_score(truth >= 0, preds >= 0)
    non_zeros = np.array([i for i, e in enumerate(truth) if e != 0])
    ex_zero_acc = accuracy_score((truth[non_zeros] > 0), (preds[non_zeros] > 0))

    preds_a7 = np.clip(preds, a_min=-3., a_max=3.)
    truth_a7 = np.clip(truth, a_min=-3., a_max=3.)
    acc_7 = multiclass_acc(preds_a7, truth_a7)

    # F1 scores. All of them are recommended by previous work.
    f1_mfn = f1_score(np.round(truth), np.round(preds), average="weighted")  # We don't use it, do we?
    f1_raven = f1_score(truth >= 0, preds >= 0, average="weighted")  # Non-negative VS. Negative
    f1_mult = f1_score((truth[non_zeros] > 0), (preds[non_zeros] > 0), average='weighted')  # Positive VS. Negative

    logging.info("\t%s mean error: %f" % (split, mae))
    logging.info("\t%s correlation: %f" % (split, cor))
    logging.info("\t%s accuracy: %f" % (split, acc))
    logging.info("\t%s accuracy 7: %f" % (split, acc_7))
    logging.info("\t%s exclude zero accuracy: %f" % (split, ex_zero_acc))
    # left and right refers to left side / right side value in Table 1 of https://arxiv.org/pdf/1911.09826.pdf
    logging.info("\t%s F1 score (raven): %f " % (split, f1_raven))  # includes zeros, Non-negative VS. Negative
    logging.info("\t%s F1 score (mult): %f " % (split, f1_mult))  # exclude zeros, Positive VS. Negative
    return mae, cor, acc, ex_zero_acc, acc_7, f1_mfn, f1_raven, f1_mult


def logSummary():
    logging.info("best epoch: %d" % gc.best.best_epoch)
    if gc.dataset == "iemocap_unaligned" or gc.dataset == "iemocap":
        for split in ["test", "valid", "test_at_valid_max"]:
            for em in gc.best.iemocap_emos:
                print("highest %s %s F1: %f" % (split, em, gc.best.max_iemocap_f1[split][em]))
                print("highest %s %s accuracy: %f" % (split, em, gc.best.max_iemocap_acc[split][em]))
    else:
        logging.info("lowest training MAE: %f" % gc.best.min_train_mae)

        logging.info("best validation epoch: %f" % gc.best.best_val_epoch)

        logging.info("best validation epoch lr: {}".format(gc.best.best_val_epoch_lr))
        logging.info("lowest validation MAE: %f" % gc.best.min_valid_mae)

        logging.info("highest validation correlation: %f" % gc.best.max_valid_cor)
        logging.info("highest validation accuracy: %f" % gc.best.max_valid_acc)
        logging.info("highest validation exclude zero accuracy: %f" % gc.best.max_valid_ex_zero_acc)
        logging.info("highest validation accuracy 7: %f" % gc.best.max_test_acc_7)
        logging.info("highest validation F1 score (raven): %f" % gc.best.max_valid_f1_raven)
        logging.info("highest validation F1 score (mfn): %f" % gc.best.max_valid_f1_mfn)
        logging.info("highest validation F1 score (mult): %f" % gc.best.max_valid_f1_mult)

        for k, v in gc.best.checkpoints_val_mae.items():
            logging.info('checkpoints {} val mae {}'.format(k, v))

def summary_to_dict():
    results = {}


    if gc.dataset == "iemocap_unaligned" or gc.dataset == "iemocap":
        for split in ["valid"]:
            for em in gc.best.iemocap_emos:
                results[f"highest {split} {em} epoch"] = gc.best.best_iemocap_epoch[split][em]
                results[f"highest {split} {em} F1"] = gc.best.max_iemocap_f1[split][em]
                results[f"highest {split} {em} accuracy"] = gc.best.max_iemocap_acc[split][em]
    else:

        results["lowest training MAE"] = gc.best.min_train_mae
        results["lowest validation MAE"] = gc.best.min_valid_mae

        results["best validation epoch"] = gc.best.best_val_epoch
        results["best validation epoch lr"] = gc.best.best_val_epoch_lr
        results["lowest validation MAE"] = gc.best.min_valid_mae

        results["highest validation correlation"] = gc.best.max_valid_cor
        results["highest validation accuracy"] = gc.best.max_valid_acc
        results["highest validation exclude zero accuracy"] = gc.best.max_valid_ex_zero_acc
        results["highest validation accuracy 7"] = gc.best.max_valid_acc_7
        results["highest validation F1 score (raven)"] = gc.best.max_valid_f1_raven
        results["highest validation F1 score (mfn)"] = gc.best.max_valid_f1_mfn
        results["highest validation F1 score (mult)"] = gc.best.max_valid_f1_mult


        for k, v in gc.best.checkpoints_val_mae.items():
            results['checkpoints {} val mae'.format(k)] = v


        for k, v in gc.best.checkpoints_val_ex_0_acc.items():
            results['checkpoints {} val ex zero acc'.format(k)] = v

    return results

def train_model(
    optimizer,
    use_gnn=True,
    exclude_vision=False,
    exclude_audio=False,
    exclude_text=False,
    average_mha=False,
    num_gat_layers=1,
    lr_scheduler=None,
    reduce_on_plateau_lr_scheduler_patience=None,
    reduce_on_plateau_lr_scheduler_threshold=None,
    multi_step_lr_scheduler_milestones=None,
    exponential_lr_scheduler_gamma=None,
    use_pe=False,
    use_prune=False
):
    assert lr_scheduler in ['reduce_on_plateau', 'exponential', 'multi_step',
                            None], 'LR scheduler can only be [reduce_on_plateau, exponential, multi_step]!'

    if gc.log_path != None:
        checkpoint_dir = os.path.join(gc.log_path, 'checkpoints')
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)

    if gc.dataset == "mosi":
        ds = MosiDataset
    elif gc.dataset == "mosi_unaligned":
        ds = MosiDatasetUnaligned
    elif gc.dataset == "mosei":
        ds = MoseiDataset
    elif gc.dataset == "mosei_unaligned":
        ds = MoseiDatasetUnaligned
    elif gc.dataset == "iemocap_unaligned":
        ds = IemocapDatasetUnaligned
    elif gc.dataset == "iemocap":
        ds = IemocapDataset
    else:
        ds = MoseiDataset

    train_dataset = ds(gc.data_path, cls="train")
    train_loader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=gc.config['batch_size'],
        shuffle=True,
        num_workers=1,
    )

    test_dataset = ds(gc.data_path, cls="test")
    test_loader = Data.DataLoader(
        dataset=test_dataset,
        batch_size=gc.config['batch_size'],
        shuffle=False,
        num_workers=1,
    )

    valid_dataset = ds(gc.data_path, cls="valid")
    valid_loader = Data.DataLoader(
        dataset=valid_dataset,
        batch_size=gc.config['batch_size'],
        shuffle=False,
        num_workers=1,
    )

    if gc.single_gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
    else:
        device = torch.device("cuda:%d" % gc.config['cuda'] if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(gc.config['cuda'])
    gc.device = device
    logging.info("running device: {}".format(device))
    gc().logParameters()

    num_aggr_nodes_per_frame = []
    if not exclude_vision:
        num_aggr_nodes_per_frame.append(gc.config['num_vision_aggr'])
    if not exclude_audio:
        num_aggr_nodes_per_frame.append(gc.config['num_audio_aggr'])
    if not exclude_text:
        num_aggr_nodes_per_frame.append(gc.config['num_text_aggr'])

    assert len(num_aggr_nodes_per_frame) != 0, "num_aggr_nodes_per_frame cannot be zero!"
    if use_gnn:
        if not average_mha:
            raise NotImplementedError('Currently only average_mha=1 is supported.')
        else:
            net = NetMTGATAverageUnalignedConcatMHA(
                num_gat_layers=num_gat_layers, use_prune=use_prune, use_pe=use_pe
            )
    else:
        raise NotImplementedError
    net.to(device)
    actual_lr_scheduler, criterion, optimizer = declare_loss_and_optimizers(
        device, exponential_lr_scheduler_gamma,
        lr_scheduler,
        multi_step_lr_scheduler_milestones, net,
        optimizer,
        reduce_on_plateau_lr_scheduler_patience,
        reduce_on_plateau_lr_scheduler_threshold,
        use_gnn
    )


    start_epoch = 0
    current_lr = 0.0
    for epoch in range(start_epoch, gc.config['epoch_num']):
        if gc.save_grad and epoch in save_epochs:
            grad_dict = {}
            update_dict = {}
        if epoch % 100 == 0:
            logSummary()

        # ======== Train Loop ========
        train_epoch_loss = 0.0
        tot_num = 0
        tot_err = 0
        tot_right = 0
        label_all = []
        output_all = []
        max_i = 0
        train_loader_length = len(train_loader)
        train_set = iter(train_loader)
        pbar = trange(train_loader_length)
        # for i in trange(train_loader_length):
        for i in pbar:
            net.train()
            optimizer.zero_grad()
            batch_update_dict = {}
            max_i = i
            words, covarep, facet, inputLen, labels = train_set.next()

            if gc.config['zero_out_video']:
                facet = torch.zeros_like(facet)
            if gc.config['zero_out_text']:
                words = torch.zeros_like(words)
            if gc.config['zero_out_audio']:
                covarep = torch.zeros_like(covarep)
            if covarep.size()[0] == 1:
                continue
            try:
                actual_batch_size, input_kwargs, labels = prepare_data_one_batch(
                    covarep, exclude_audio, exclude_text,
                    exclude_vision, facet, inputLen,
                    labels, words, device
                )
            except ValueError:
                continue
            try:
                outputs = net(**input_kwargs)
            except Exception as e:
                tb = traceback.format_exc()
                print(tb)
                pass

            output_all.extend(outputs.tolist())
            label_all.extend(labels.tolist())
            if gc.dataset in ['iemocap', 'iemocap_unaligned']:
                outputs = outputs.view(-1, 2)
                labels = labels.view(-1)
            else:
                err = torch.sum(torch.abs(outputs - labels))
                tot_right += torch.sum(torch.eq(torch.sign(labels), torch.sign(outputs)))
                tot_err += err

            tot_num += actual_batch_size

            loss = criterion(outputs, labels)
            if gc.config['use_loss_norm']:
                loss = loss / torch.abs(loss.detach())
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=gc.config['max_grad'], norm_type=inf)
            if gc.save_grad and epoch in save_epochs:
                for name, param in net.named_parameters():
                    if param.grad is None:
                        continue
                    try:
                        if i == 0:
                            grad_dict[name] = param.grad.detach().cpu().numpy()
                        else:
                            grad_dict[name] = grad_dict[name] + np.abs(param.grad.detach().cpu().numpy())
                        assert (name not in batch_update_dict)
                        batch_update_dict[name] = param.data.detach().cpu().numpy()
                    except:
                        import pdb
                        pdb.set_trace()
            optimizer.step()
            if gc.save_grad and epoch in save_epochs:
                for name, param in net.named_parameters():
                    if param.grad is None:
                        continue
                    if i == 0:
                        update_dict[name] = np.abs(batch_update_dict[name] - param.data.detach().cpu().numpy())
                    else:
                        update_dict[name] += np.abs(batch_update_dict[name] - param.data.detach().cpu().numpy())
            train_epoch_loss += loss.item() * actual_batch_size
            pbar.set_description("GPU Mem: {:.2f}/{:.2f} GiB, LR: {}".format(
                torch.cuda.max_memory_allocated(device) / (2. ** 30),
                torch.cuda.get_device_properties(device).total_memory / (2. ** 30),
                optimizer.param_groups[0]['lr']
            ))

            del loss
            del outputs
            torch.cuda.empty_cache()


        logging.info(f'\nEpoch {epoch}, train loss: {train_epoch_loss / tot_num:.3f}')

        if gc.dataset == "iemocap_unaligned" or gc.dataset == "iemocap":
            train_f1, train_acc = eval_iemocap('train', output_all, label_all)
            for em in gc.best.iemocap_emos:
                if train_f1[em] > gc.best.max_iemocap_f1['train'][em]:
                    gc.best.best_iemocap_epoch['train'][em] = epoch
                    gc.best.max_iemocap_f1['train'][em] = train_f1[em]
                if train_acc[em] > gc.best.max_iemocap_acc['train'][em]:
                    gc.best.max_iemocap_acc['train'][em] = train_acc[em]

        else:
            train_mae = tot_err / tot_num
            train_acc = float(tot_right) / tot_num
            logging.info("\ttrain mean error: %f" % train_mae)
            logging.info("\ttrain accuracy: %f" % train_acc)
            if train_mae < gc.best.min_train_mae:
                gc.best.min_train_mae = train_mae.item()
            if train_acc > gc.best.max_train_acc:
                gc.best.max_train_acc = train_acc

        if gc.save_grad and epoch in save_epochs:
            grad_f = h5py.File(os.path.join(gc.model_path, '%s_grad_%s_%d.hdf5' % (gc.dataset, config_name, epoch)))
            update_f = h5py.File(os.path.join(gc.model_path, '%s_update_%s_%d.hdf5' % (gc.dataset, config_name, epoch)))
            for name in grad_dict.keys():
                grad_avg = grad_dict[name] / (max_i + 1)
                grad_f.create_dataset(name, data=grad_avg)
                update_avg = update_dict[name] / (max_i + 1)
                update_f.create_dataset(name, data=update_avg)
            grad_f.close()
            update_f.close()

        if False and gc.log_path is not None:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpiont_epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_arch': net,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, checkpoint_path)

        # ======== Validation Loop ========
        with torch.no_grad():
            net.eval()

            valid_label_all = []
            valid_output_all = []
            valid_loss = 0.0
            count = 0
            for data in valid_loader:
                net.eval()
                words, covarep, facet, inputLen, labels = data

                if gc.config['zero_out_video']:
                    facet = torch.zeros_like(facet)
                if gc.config['zero_out_text']:
                    words = torch.zeros_like(words)
                if gc.config['zero_out_audio']:
                    covarep = torch.zeros_like(covarep)

                if covarep.size()[0] == 1:
                    continue
                try:
                    actual_batch_size, input_kwargs, labels = prepare_data_one_batch(covarep,
                                                                                     exclude_audio,
                                                                                     exclude_text,
                                                                                     exclude_vision,
                                                                                     facet, inputLen,
                                                                                     labels, words, device)
                except ValueError:
                    continue
                try:
                    outputs = net(**input_kwargs)
                except:
                    pass

                valid_output_all.extend(outputs.data.cpu().tolist())
                valid_label_all.extend(labels.data.cpu().tolist())
                if gc.dataset == "iemocap_unaligned" or gc.dataset == "iemocap":
                    outputs = outputs.view(-1, 2)
                    labels = labels.view(-1)
                valid_loss += criterion(outputs, labels).item()
                count += 1

            # Evaluate metrics
            if gc.dataset == "iemocap_unaligned" or gc.dataset == "iemocap":
                valid_loss /= count
                valid_f1, valid_acc = eval_iemocap('valid', valid_output_all, valid_label_all)
            else:
                valid_mae, valid_cor, valid_acc, valid_ex_zero_acc, valid_acc_7, valid_f1_mfn, valid_f1_raven, valid_f1_mult = eval_mosi_mosei(
                    'valid', valid_output_all, valid_label_all)

            # Update best metrics
            if gc.dataset == "iemocap_unaligned" or gc.dataset == "iemocap":
                # TODO: log the valid metrics as opposed to test
                for em in gc.best.iemocap_emos:
                    if valid_f1[em] > gc.best.max_iemocap_f1['valid'][em]:
                        gc.best.best_iemocap_epoch['valid'][em] = epoch
                        gc.best.max_iemocap_f1['valid'][em] = valid_f1[em]
                        gc.best.best_epoch = epoch
                    if valid_acc[em] > gc.best.max_iemocap_acc['valid'][em]:
                        gc.best.max_iemocap_acc['valid'][em] = valid_acc[em]
            else:
                # log the best validation performance (valid MAE)
                if valid_mae < gc.best.min_valid_mae:
                    gc.best.min_valid_mae = valid_mae
                    gc.best.best_val_epoch = epoch
                    gc.best.best_val_epoch_lr = current_lr
                    best_model = True

                if valid_cor > gc.best.max_valid_cor:
                    gc.best.max_valid_cor = valid_cor
                if valid_acc > gc.best.max_valid_acc:
                    gc.best.max_valid_acc = valid_acc
                if valid_ex_zero_acc > gc.best.max_valid_ex_zero_acc:
                    gc.best.max_valid_ex_zero_acc = valid_ex_zero_acc
                if valid_f1_raven > gc.best.max_valid_f1_raven:
                    gc.best.max_valid_f1_raven = valid_f1_raven
                if valid_f1_mfn > gc.best.max_valid_f1_mfn:
                    gc.best.max_valid_f1_mfn = valid_f1_mfn
                if valid_f1_mult > gc.best.max_valid_f1_mult:
                    gc.best.max_valid_f1_mult = valid_f1_mult
                if valid_acc_7 > gc.best.max_valid_acc_7:
                    gc.best.max_valid_acc_7 = valid_acc_7

                # log the best valid mae and the corresponding validation performances
                if valid_mae < gc.best.min_valid_mae:
                    gc.best.min_valid_mae = valid_mae
                    gc.best.best_val_epoch = epoch
                    gc.best.best_val_epoch_lr = current_lr
                    best_model = True

                    if gc.config['save_best_model']:
                        checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
                        logging.info(f"Saving best model to {checkpoint_path}")
                        torch.save({
                            'best_epoch': epoch,
                            # 'model_arch': net,
                            'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()
                        }, checkpoint_path)

                # log the checkpoints for val
                if epoch in gc.config['checkpoints']:
                    gc.best.checkpoints_val_mae[epoch] = valid_mae
                    gc.best.checkpoints_val_ex_0_acc[epoch] = valid_ex_zero_acc

            if actual_lr_scheduler is not None:
                if isinstance(actual_lr_scheduler, ReduceLROnPlateau):
                    if gc.dataset == "iemocap_unaligned" or gc.dataset == "iemocap":  # IEMOCAP does not have MAE
                        actual_lr_scheduler.step(valid_loss)
                    else:
                        actual_lr_scheduler.step(valid_mae)
                else:
                    actual_lr_scheduler.step()
                current_lr = actual_lr_scheduler._last_lr

    logSummary()
    return summary_to_dict()

def declare_loss_and_optimizers(
    device, exponential_lr_scheduler_gamma, lr_scheduler,
    multi_step_lr_scheduler_milestones, net, optimizer,
    reduce_on_plateau_lr_scheduler_patience, reduce_on_plateau_lr_scheduler_threshold,
    use_gnn
):
    if gc.config['loss_type'] == 'mse':
        criterion = nn.MSELoss()
    elif gc.config['loss_type'] == 'l1':
        criterion = nn.L1Loss()
    elif gc.config['loss_type'] == 'sl1':
        criterion = nn.SmoothL1Loss()
    else:
        raise NotImplementedError
    if gc.dataset == "iemocap_unaligned" or gc.dataset == "iemocap":
        if gc.config['use_iemocap_inverse_sample_count_ce_loss']:
            criterion = IEMOCAPInverseSampleCountCELoss()
            criterion.to(device)
        else:
            criterion = nn.CrossEntropyLoss()
    if optimizer == "sgd":
        if use_gnn:
            optimizer = optim.SGD([
                {'params': list(net.dgnn.parameters()) + list(net.finalW.parameters())},
                # {'params': net.gru.parameters(), 'lr': gc.config['gru_lr']}
            ],
                lr=gc.config['global_lr'],
                momentum=gc.config['momentum'],
                weight_decay=gc.config['weight_decay'])
        else:
            optimizer = optim.SGD([
                {'params': list(net.vision_fc.parameters()) +
                           list(net.text_fc.parameters()) +
                           list(net.audio_fc.parameters()) +
                           list(net.finalW.parameters())
                 },
                {'params': net.gru.parameters(), 'lr': gc.config['gru_lr']}
            ],
                lr=gc.config['global_lr'],
                momentum=gc.config['momentum'],
                weight_decay=gc.config['weight_decay'])
    elif optimizer == "adam":
        if use_gnn:
            optimizer = optim.Adam([
                {'params': list(net.dgnn.parameters()) + list(net.finalW.parameters())},
                # {'params': net.gru.parameters(), 'lr': gc.config['gru_lr']}
            ],
                lr=gc.config['global_lr'],
                betas=(gc.config['beta1'], gc.config['beta2']),
                eps=gc.config['eps'],
                weight_decay=gc.config['weight_decay'])
        else:
            optimizer = optim.Adam([
                {'params': list(net.vision_fc.parameters()) +
                           list(net.text_fc.parameters()) +
                           list(net.audio_fc.parameters()) +
                           list(net.finalW.parameters())
                 },
                {'params': net.gru.parameters(), 'lr': gc.config['gru_lr']}
            ],
                lr=gc.config['global_lr'],
                betas=(gc.config['beta1'], gc.config['beta2']),
                eps=gc.config['eps'],
                weight_decay=gc.config['weight_decay'])
    elif optimizer == "adamw":
        if use_gnn:
            optimizer = optim.AdamW([
                {'params': list(net.dgnn.parameters()) + list(net.finalW.parameters())},
                # {'params': net.gru.parameters(), 'lr': gc.config['gru_lr']}
            ],
                lr=gc.config['global_lr'],
                weight_decay=gc.config['weight_decay'],
                betas=(gc.config['beta1'], gc.config['beta2']),
                eps=gc.config['eps'])
        else:
            optimizer = optim.AdamW([
                {'params': list(net.vision_fc.parameters()) +
                           list(net.text_fc.parameters()) +
                           list(net.audio_fc.parameters()) +
                           list(net.finalW.parameters())
                 },
                {'params': net.gru.parameters(), 'lr': gc.config['gru_lr']}
            ],
                lr=gc.config['global_lr'],
                weight_decay=gc.config['weight_decay'],
                betas=(gc.config['beta1'], gc.config['beta2']),
                eps=gc.config['eps'])
    else:
        logging.info("Unsupported optimizer {}".format(optimizer))
        sys.exit()
    if lr_scheduler is not None:
        if lr_scheduler == 'reduce_on_plateau':
            assert isinstance(reduce_on_plateau_lr_scheduler_patience, int)
            assert isinstance(reduce_on_plateau_lr_scheduler_threshold, float)
            actual_lr_scheduler = ReduceLROnPlateau(optimizer,
                                                    factor=0.5,
                                                    patience=reduce_on_plateau_lr_scheduler_patience,
                                                    threshold=reduce_on_plateau_lr_scheduler_threshold)
        elif lr_scheduler == 'exponential':
            assert exponential_lr_scheduler_gamma is not None
            actual_lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=exponential_lr_scheduler_gamma)
        elif lr_scheduler == 'multi_step':
            assert multi_step_lr_scheduler_milestones is not None
            milestones = [int(step) for step in multi_step_lr_scheduler_milestones.split(',')]
            actual_lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones)
        else:
            raise NotImplementedError
    return actual_lr_scheduler, criterion, optimizer

def prepare_data_one_batch(
    covarep, exclude_audio, 
    exclude_text, exclude_vision, 
    facet, inputLen, 
    labels, words, device
):
    words, covarep, facet, inputLen, labels = words.to(device), covarep.to(device), facet.to(
        device), inputLen.to(device), labels.to(device)
    input_kwargs = {}
    if not exclude_text:
        t_mask = torch.abs(words).sum(-1) != 0
    if not exclude_vision:
        v_mask = torch.abs(facet).sum(-1) != 0
    if not exclude_audio:
        a_mask = torch.abs(covarep).sum(-1) != 0
    if torch.sum(t_mask) == 0 and torch.sum(v_mask) == 0 and torch.sum(a_mask) == 0:
        logging.info("encountered an all zero mask")
        raise ValueError
    # exclude all zero dataset
    words_exclude = torch.sum(t_mask, -1) == 0
    facet_exclude = torch.sum(v_mask, -1) == 0
    covarep_exclude = torch.sum(a_mask, -1) == 0
    exclude_inds = words_exclude & facet_exclude & covarep_exclude
    words = words[~exclude_inds]
    t_mask = t_mask[~exclude_inds]
    facet = facet[~exclude_inds]
    v_mask = v_mask[~exclude_inds]
    covarep = covarep[~exclude_inds]
    a_mask = a_mask[~exclude_inds]
    labels = labels[~exclude_inds]
    input_kwargs['text'] = words
    input_kwargs['vision'] = facet
    input_kwargs['audio'] = covarep

    input_kwargs['t_mask'] = t_mask
    input_kwargs['v_mask'] = v_mask
    input_kwargs['a_mask'] = a_mask

    actual_batch_size = covarep.shape[0]
    return actual_batch_size, input_kwargs, labels


def eval_model(resume_pt):
    if gc.dataset == "mosi":
        ds = MosiDataset
    elif gc.dataset == "mosi_unaligned":
        ds = MosiDatasetUnaligned
    elif gc.dataset == "iemocap_unaligned":
        ds = IemocapDatasetUnaligned
    elif gc.dataset == "iemocap":
        ds = IemocapDataset
    else:
        ds = MoseiDataset

    test_dataset = ds(gc.data_path, cls="test")
    test_loader = Data.DataLoader(
        dataset=test_dataset,
        batch_size=gc.config['batch_size'],
        shuffle=False,
        num_workers=1,
    )

    if gc.single_gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
    else:
        device = torch.device("cuda:%d" % gc.config['cuda'] if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(gc.config['cuda'])
    gc.device = device
    logging.info("running device: {}".format(device))
    gc().logParameters()

    checkpoint = torch.load(resume_pt)
    # net = checkpoint['model_arch']
    net = NetMTGATAverageUnalignedConcatMHA(num_gat_layers=gc.config['num_gat_layers'],
                                            use_prune=gc.config['use_prune'],
                                            use_pe=gc.config['use_pe'])
    net.load_state_dict(checkpoint['model_state_dict'])
    net.to(device)

    with torch.no_grad():
        net.eval()
        test_label_all = []
        test_output_all = []
        for data in test_loader:
            words, covarep, facet, inputLen, labels = data
            if covarep.size()[0] == 1:
                continue
            words, covarep, facet, inputLen, labels = words.to(device), covarep.to(device), facet.to(
                device), inputLen.to(device), labels.to(device)

            if gc.config['zero_out_video']:
                facet = torch.zeros_like(facet)
            if gc.config['zero_out_text']:
                words = torch.zeros_like(words)
            if gc.config['zero_out_audio']:
                covarep = torch.zeros_like(covarep)

            if covarep.size()[0] == 1:
                continue
            try:
                actual_batch_size, input_kwargs, labels = prepare_data_one_batch(covarep,
                                                                                 False,
                                                                                 False,
                                                                                 False,
                                                                                 facet, inputLen,
                                                                                 labels, words, device)
            except ValueError:
                continue
            outputs = net(**input_kwargs)
            test_output_all.extend(outputs.tolist())
            test_label_all.extend(labels.tolist())

        if gc.dataset == "mosi" or gc.dataset == "mosi_unaligned":
            test_mae, test_cor, test_acc, test_ex_zero_acc, test_acc_7, test_f1_mfn, test_f1_raven, test_f1_mult = \
                eval_mosi_mosei('test', test_output_all, test_label_all)
        elif gc.dataset == "iemocap_unaligned" or gc.dataset == "iemocap":
            test_acc, test_f1 = eval_iemocap('test', test_output_all, test_label_all)

        else:
            test_mae, test_cor, test_acc, test_ex_zero_acc, test_f1_raven, test_f1_mult, test_cor, test_acc_5, test_acc_7, test_f1_happy, test_weighted_acc_happy, test_f1_sad, test_weighted_acc_sad, test_f1_anger, test_weighted_acc_anger, test_f1_fear, test_weighted_acc_fear, test_f1_disgust, test_weighted_acc_disgust, test_f1_surprise, test_weighted_acc_surprise = \
                eval_mosi_mosei('test', test_output_all, test_label_all)