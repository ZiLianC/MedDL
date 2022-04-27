import os
import time
import shutil
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F 
from torch.cuda.amp import GradScaler, autocast
from tensorboardX import SummaryWriter
from utils.utils import distributed_all_gather
import torch.utils.data.distributed
from monai.data import decollate_batch
from utils.valid_utils import AverageMeter,to_one_hot_3d,to_one_hot_3d_target

'''func for training'''
def train_epoch(model,
                loader,
                optimizer,
                scaler,
                epoch,
                loss_func,
                loss_con,
                args):
    # set in train mode
    model.train()
    start_time = time.time()
    # calculate losses
    run_loss = AverageMeter()
    # DS coeffis
    alpha = 0.4
    if epoch % 30 == 0: alpha *= 0.8
    for idx, batch_data in tqdm(enumerate(loader)):
        # clean cuda cached useless grad graph
        torch.cuda.empty_cache()
        # try batch_data is a list to select ways of loading. for compatibility
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data['image'], batch_data['label']
        # set data downstream to cuda 
        images, target = data.cuda(args.rank), target.cuda(args.rank)
        # set non grad for training
        for param in model.parameters(): param.grad = None
        # cuda opt
            # training and loss calculation
        logits = model(images)
        loss = loss_func(logits,target)
            #loss = loss_func(logits[2], target)+alpha*(loss_func(logits[1], target)+loss_func(logits[0], target))
        # back propagation with cuda opt.
            # normal bp
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        run_loss.update(loss.item(), n=args.batch_size)
        # print data
        if args.rank == 0:
            print('Epoch {}/{} {}/{}'.format(epoch, args.max_epochs, idx, len(loader)),
                  'loss: {:.4f}'.format(run_loss.avg),
                  'time {:.2f}s'.format(time.time() - start_time))
        torch.cuda.empty_cache()
        start_time = time.time()
    for param in model.parameters() : param.grad = None
    
    return run_loss.avg

def val_epoch(model,
              loader,
              epoch,
              acc_func,
              args,
              model_inferer=None,
              post_label=None,
              post_pred=None):
    # evalution mode
    model.eval()
    start_time = time.time()
    # standard evaluation paradigm
    with torch.no_grad():
        y_val=[]
        y_pred=[]
        for idx, batch_data in enumerate(loader):
            torch.cuda.empty_cache()   
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data['image'], batch_data['label']
            data, target = data.cuda(args.rank), target.cuda(args.rank)
            logits = model(data)
            if not logits.is_cuda:
                target = target.cpu()
            torch.cuda.empty_cache()
            # calulate metric.
            logits=F.softmax(logits,dim=1)
            y_val.append(target.cpu().item())
            y_pred.append(logits.argmax(dim=1).cpu().item())
            print(logits.argmax(dim=1).cpu().item())
            torch.cuda.empty_cache()
            start_time = time.time()
        acc = acc_func(y_val,y_pred)
        print(acc)
    return acc

def save_checkpoint(model,
                    epoch,
                    args,
                    filename='model',
                    best_acc=0,
                    optimizer=None,
                    scheduler=None):
    # save state dict & best epoch & epoch & optimizer & scheduler for checkpoint reload
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {
            'epoch': epoch,
            'best_acc': best_acc,
            'state_dict': state_dict
            }
    if optimizer is not None:
        save_dict['optimizer'] = optimizer.state_dict()
    if scheduler is not None:
        save_dict['scheduler'] = scheduler.state_dict()
    filename_state_dict = filename + "_state_dict.pth"
    filename_state_dict=os.path.join(args.logdir, filename_state_dict)
    torch.save(save_dict, filename_state_dict)
    
    # save a wrap for convenience
    filename_model = filename + "_model.pth"
    filename_model=os.path.join(args.logdir, filename_model)
    torch.save(model,filename_model)
    print('Saving checkpoint', filename)


'''core training'''
def run_training(model,
                 train_loader,
                 val_loader,
                 optimizer,
                 loss_func,
                 loss_con,
                 acc_func,
                 args,
                 model_inferer=None,
                 scheduler=None,
                 start_epoch=0,
                 post_label=None,
                 post_pred=None
                 ):
    writer = None
    # tensorboard logging
    if args.logdir is not None and args.rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)
        if args.rank == 0: print('Writing Tensorboard logs to ', args.logdir)
    val_acc_max = 0.
    
    # epoch iteration
    for epoch in range(start_epoch, args.max_epochs):
        torch.cuda.empty_cache()
        # distrib. ln. setting
        print(args.rank, time.ctime(), 'Epoch:', epoch)
        epoch_time = time.time()
        # training
        train_loss = train_epoch(model,
                                 train_loader,
                                 optimizer,
                                 scaler=None,
                                 epoch=epoch,
                                 loss_func=loss_func,
                                 loss_con = loss_con,
                                 args=args)
        if args.rank == 0:
            print('Final training  {}/{}'.format(epoch, args.max_epochs - 1), 'loss: {:.4f}'.format(train_loss),
                  'time {:.2f}s'.format(time.time() - epoch_time))

        # write loss to tbx
        if args.rank==0 and writer is not None:
            writer.add_scalar('train_loss', train_loss, epoch)
        b_new_best = False

        # save model & weight in selected epochs.
        if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
            save_checkpoint(model,
                                epoch,
                                args,
                                best_acc=val_acc_max,
                                filename='model_latest')
        
        # evaluate model & weight in selected epochs.
        if (epoch+1) % args.val_every == 0:
            torch.cuda.empty_cache()
            epoch_time = time.time()
            val_avg_acc = val_epoch(model,
                                    val_loader,
                                    epoch=epoch,
                                    acc_func=acc_func,
                                    model_inferer=model_inferer,
                                    args=args,
                                    post_label=post_label,
                                    post_pred=post_pred)
            # write to tbx
            if args.rank == 0:
                print('Final validation  {}/{}'.format(epoch, args.max_epochs - 1),
                      'acc', val_avg_acc, 'time {:.2f}s'.format(time.time() - epoch_time))
                if writer is not None:
                    writer.add_scalar('val_acc', val_avg_acc, epoch)
                if val_avg_acc > val_acc_max:
                    print('new best ({:.6f} --> {:.6f}). '.format(val_acc_max, val_avg_acc))
                    val_acc_max = val_avg_acc
                    b_new_best = True
                    if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                        save_checkpoint(model, epoch, args,filename="model_best",
                                        best_acc=val_acc_max,
                                        optimizer=optimizer,
                                        scheduler=scheduler)
        # update scheduler epoch-wise
        if scheduler is not None:
            scheduler.step()

    print('Training Finished !, Best Accuracy: ', val_acc_max)

    return val_acc_max

