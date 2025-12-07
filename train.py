import sys, os


curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(os.path.split(rootPath)[0])

from os import path
d = path.dirname(__file__)
parent_path = os.path.dirname(d)
sys.path.append(parent_path)

from ICTA2Net.DETRIS.model import build_segmenter
import torch
import torch.nn as nn
import numpy as np
from ICTA2Net.dataset.dataset import AVADataset, AVADataset_test

from ICTA2Net.utils.utils import AverageMeter, RankingLoss, RankNetLoss, RegRankLoss, ContrastiveLoss, GroupContrastiveLoss
from ICTA2Net.utils import option
from tqdm import tqdm

from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from tensorboardX import SummaryWriter
from sklearn.metrics import accuracy_score

from timm.models import create_model
import torch
import random


f = open('/home/cjg/workSpace/AVA_image_sort_v2/ICTA2Net/logs/log.txt', 'w')
opt = option.init()
opt.device = torch.device("cuda:{}".format(opt.gpu_id))

def create_data_part(opt):
    train_csv_path = os.path.join(opt.train_csv_path, 'train_8.csv')
    
    test_csv_path = os.path.join(opt.test_csv_path, 'ICTAA-GP.csv')
    validate_csv_path = os.path.join(opt.test_csv_path, 'ICTAA-HP.csv')

    test_csv_gt = os.path.join(opt.test_csv_path, 'ICTAA-GF.csv')
    val_csv_gt = os.path.join(opt.test_csv_path, 'ICTAA-HF.csv')

    root_dir = opt.image_path_G
    root_test_dir = opt.image_path_G
    root_validate_dir = opt.image_path_H

    train_ds = AVADataset(train_csv_path, root_dir, if_train=True, ablate_text=opt.ablate_text)

    test_ds = AVADataset_test(test_csv_path, root_test_dir, if_train=False, ablate_text=opt.ablate_text)
    val_ds = AVADataset_test(validate_csv_path, root_validate_dir, if_train=False, ablate_text=opt.ablate_text) 
    
    test_gt_ds = AVADataset_test(test_csv_gt, root_test_dir, if_train=False, ablate_text=opt.ablate_text)
    val_gt_ds = AVADataset_test(val_csv_gt, root_validate_dir, if_train=False, ablate_text=opt.ablate_text)

    train_loader = DataLoader(train_ds, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=True)

    test_loader = DataLoader(test_ds, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=False)

    test_gt_loader = DataLoader(test_gt_ds, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=False)
    val_gt_loader = DataLoader(val_gt_ds, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=False)

    return train_loader, test_loader, val_loader, test_gt_loader, val_gt_loader


def train(opt, model, loader, optimizer, criterion, bce_criterion, mse_criterion,loss_fn, contrastive_loss, writer=None, global_step=None, name=None):
    model.train()
    train_losses = AverageMeter()
    for idx, (img1, img2, text1, text2, score1, score2, labels, image_name1, image_name2, group_ids) in enumerate(tqdm(loader)):
        img1, img2 = img1.to(opt.device), img2.to(opt.device)
        text1 = text1.to(opt.device)
        score1, score2 = score1.to(opt.device).to(torch.float32), score2.to(opt.device).to(torch.float32)
        labels = labels.view(-1,1).to(opt.device)
        group_ids = group_ids.to(opt.device)

        # get model prediction scores for two images
        pred1, pred2, proj_vis1, proj_txt1, proj_vis2, proj_txt2 = model(img1, img2, text1)
        loss_reg_rank, loss_reg, loss_rank = loss_fn(
            y_pred=(pred1, pred2),
            y_true=(score1, score2)
        )

        # compute contrastive loss
        loss_contrastive = contrastive_loss((proj_vis1, proj_vis2),(proj_txt1, proj_txt2), group_ids)
        # compute rank loss
        loss_ranknet = criterion(pred1, pred2, labels)

        # The images are concatenated and scored; softmax is then used to obtain the probability that each image is "considered better".
        scores_pred = torch.cat((pred1, pred2), dim=1)  # shape: (B, 2)
        prob_pred = torch.softmax(scores_pred * 5, dim=1)[:, 0:1]  # probability that pred1 is better

        # Convert the label to the probability value corresponding to prob_pred.（1 represents pred1 being better, 0 represents pred2 being better）
        # The labels were originally -1, 0, 1; they are converted to 0, 0.5, 1 to represent probabilities.
        prob_label = 0.5 * (1 + labels.float())  # shape: (B, 1)
        
        # BCE and MSE loss calculation
        bce_loss = bce_criterion(prob_pred, prob_label)
        mse_loss = mse_criterion(prob_pred, prob_label)

        # merge loss
        reg = 0.0
        rank = 1.0
        ranknet = 0.0
        bce = 0.0 
        mse = 1.0
        contra = 1.0
        
        loss = ranknet * loss_ranknet + bce * bce_loss + mse * mse_loss + rank  * loss_rank + reg * loss_reg + contra * loss_contrastive

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_losses.update(loss.item(), img1.size(0))

        if writer is not None:
            writer.add_scalar(f"{name}/train_loss.avg", train_losses.avg, global_step=global_step + idx)

    return train_losses.avg, loss_contrastive



# calc validation metrics
def validate(opt, model, loader, criterion, bce_criterion, mse_criterion,loss_fn, contrastive_loss, writer=None, global_step=None, name=None):
    model.eval()
    validate_losses = AverageMeter()
    true_labels = []
    pred_labels = []
    confidence_list = []

    for idx, (img1, img2, text1, text2, score1, score2, labels, image_name1, image_name2, confidence, gropu_ids) in enumerate(tqdm(loader)):
        img1, img2 = img1.to(opt.device), img2.to(opt.device)
        text1 = text1.to(opt.device)
        score1, score2 = score1.to(opt.device).to(torch.float32), score2.to(opt.device).to(torch.float32)
        confidence = confidence.to(opt.device).float().view(-1,1)
        gropu_ids = gropu_ids.to(opt.device)

        with torch.no_grad():
            # get model prediction scores for two images
            pred1, pred2, proj_vis1, proj_txt1, proj_vis2, proj_txt2 = model(img1, img2, text1, gropu_ids)

            scores = torch.cat((pred1, pred2), dim=1)  # shape: (B, 2)
            prob_pred1 = torch.softmax(scores * 5, dim=1)[:, 0:1]  # the probability that pred1 is better


            pred_scores = torch.where(prob_pred1 > 0.5, torch.ones_like(prob_pred1),
                            torch.where(prob_pred1 < 0.5, -torch.ones_like(prob_pred1), torch.zeros_like(prob_pred1)))

            # pred_scores = torch.where(pred1 > pred2, torch.ones_like(pred1), torch.where(pred1 < pred2, -torch.ones_like(pred1), torch.zeros_like(pred1)))
            true_scores = labels.view(-1,1).to(opt.device)

            loss_reg_rank, loss_reg, loss_rank = loss_fn(
                y_pred=(pred1, pred2),
                y_true=(score1, score2)
            )

            loss_contrastive = contrastive_loss((proj_vis1, proj_vis2),(proj_txt1, proj_txt2), gropu_ids)

            # calculate ranknet loss
            loss_ranknet = criterion(pred1, pred2, true_scores)

            pred = 0.5 * (1 + pred_scores)
            labels = 0.5 * (1 + true_scores)
            # calculate BCE loss
            bce_loss = bce_criterion(pred, labels.float())

            # calculate MSE loss
            mse_loss = mse_criterion(pred, labels.float())

            # merge loss
            rank = 1.0
            reg = 0.0
            ranknet =  0.0
            bce = 0.0  
            mse = 1.0  
            contra = 1.0
            loss = ranknet * loss_ranknet + bce * bce_loss + mse * mse_loss + rank * loss_rank + reg * loss_reg + contra * loss_contrastive

            validate_losses.update(loss.item(), img1.size(0))

            # put results into list for metric calculation
            true_labels.extend(true_scores.cpu().numpy().flatten())
            pred_labels.extend(pred_scores.cpu().numpy().flatten())
            confidence_list.extend(confidence.cpu().numpy().flatten())

        if writer is not None:
            writer.add_scalar(f"{name}/val_loss.avg", validate_losses.avg, global_step=global_step + idx)

    # convert to numpy array for metric calculation
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    confidence_list = np.array(confidence_list).flatten()

    # calculate SRCC and PLCC
    srcc, _ = spearmanr(true_labels, pred_labels)
    plcc, _ = pearsonr(true_labels, pred_labels)

    # calculate accuracy
    # accuracy = accuracy_score(true_labels, pred_labels)
    correct = (true_labels == pred_labels).astype(np.float32)
    weighted_correct = (correct * confidence_list).sum()
    total_confidence = confidence_list.sum()
    weighted_accuracy = weighted_correct / total_confidence if total_confidence > 0 else 0.0


    # print(f"Weighted Accuracy: {weighted_accuracy:.4f}, SRCC: {srcc:.4f}, PLCC: {plcc:.4f}")

    return validate_losses.avg, weighted_accuracy, srcc, plcc


def creat_over_locl_midel(opt):
    model = create_model(
        opt.model,
        pretrained=opt.pretrained,
        num_classes=opt.num_classes,
        drop_rate=opt.drop,
        drop_path_rate=opt.drop_path,
        # img_size=opt.input_size,
        use_checkpoint=opt.ckpt_stg if opt.grad_checkpoint else [0] * 4,
    )
    return model

def start_train(opt):
    # print parameter
    print("***************************************Initialization parameter***************************************")
    print("batch_size: {}".format(opt.batch_size))
    print("epochs: {}".format(opt.num_epoch))
    print("images_path_G: {}".format(opt.image_path_G))
    print("images_path_H: {}".format(opt.image_path_H))
    print("train_csv_path: {}".format(opt.train_csv_path))
    print("test_csv_path: {}".format(opt.test_csv_path))
    print("model_save_path: {}".format(opt.model_save_path))
    print("cuda: {}".format(torch.cuda.is_available()))
    print("ablate_text: {}".format(opt.ablate_text))
    # data, model, optimizer, criterion, writer
    train_loader,  test_loader, val_loader, test_gt_loader, val_gt_loader = create_data_part(opt)
    model, param_list = build_segmenter(opt)
    optimizer = torch.optim.Adam(param_list, lr=opt.init_lr, weight_decay=0.0001)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-7)
    model = model.to(opt.device)
    criterion = RankNetLoss().to(opt.device)
    bce_loss = nn.BCELoss().to(opt.device)
    mse_loss = nn.MSELoss().to(opt.device)
    loss_fn = RegRankLoss(margin=0.02).to(opt.device)

    contrastive_loss = GroupContrastiveLoss(temperature=0.07)

    writer = SummaryWriter(log_dir=os.path.join(opt.experiment_dir_name, 'logs'))

    # whether continue training
    start_epoch = 0
    if opt.resume == True:
        checkpoint = torch.load(opt.checkpoint_path)
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print("**************************************resume training from epoch {}**************************************".format(start_epoch))
        print("Missing keys (new modules not in checkpoint):", missing_keys)
        print("Unexpected keys (checkpoint has but model doesn't):", unexpected_keys)
    best_test_acc = 0
    best_val_acc = 0
    best_both_acc = 0
    best_test_acc_gt = 0
    best_val_acc_gt = 0
    best_test_model_name = None
    best_val_model_name = None
    best_both_model_name = None
    best_test_gt_model_name = None
    best_val_gt_model_name = None
    # start training
    for e in range(opt.num_epoch - start_epoch):
        e = e + start_epoch
        print(f"Learning rate for epoch {e}: {optimizer.param_groups[0]['lr']}")
        print("*****************************************Training epoch {}*****************************************".format(e))
        train_loss, contra_loss = train(opt,model=model, loader=train_loader, optimizer=optimizer, criterion=criterion, bce_criterion=bce_loss, mse_criterion=mse_loss,loss_fn=loss_fn, contrastive_loss = contrastive_loss,
                           writer=writer, global_step=len(train_loader) * e,
                           name=f"{opt.experiment_dir_name}_by_batch")

        print("*****************************************Testing epoch {}*****************************************".format(e))
        test_loss, tacc, tsrcc, tplcc = validate(opt, model=model, loader=test_loader, criterion=criterion, bce_criterion=bce_loss, mse_criterion=mse_loss,loss_fn=loss_fn, contrastive_loss = contrastive_loss,
                                               writer=writer, global_step=len(test_loader) * e,
                                               name=f"{opt.experiment_dir_name}_by_batch")

        print("*****************************************Validation epoch {}*****************************************".format(e))
        val_loss, vacc, vsrcc, vplcc = validate(opt, model=model, loader=val_loader, criterion=criterion, bce_criterion=bce_loss, mse_criterion=mse_loss,loss_fn=loss_fn, contrastive_loss = contrastive_loss,
                                               writer=writer, global_step=len(val_loader) * e,
                                               name=f"{opt.experiment_dir_name}_by_batch")

        print("*****************************************Testing GT epoch {}*****************************************".format(e))
        test_loss_gt, tacc_gt, tsrcc_gt, tplcc_gt = validate(opt, model=model, loader=test_gt_loader, criterion=criterion, bce_criterion=bce_loss, mse_criterion=mse_loss,loss_fn=loss_fn, contrastive_loss = contrastive_loss,
                                               writer=writer, global_step=len(test_gt_loader) * e,
                                               name=f"{opt.experiment_dir_name}_by_batch")

        print("*****************************************Validation GT epoch {}*****************************************".format(e))
        val_loss_gt, vacc_gt, vsrcc_gt, vplcc_gt = validate(opt, model=model, loader=val_gt_loader, criterion=criterion, bce_criterion=bce_loss, mse_criterion=mse_loss,loss_fn=loss_fn, contrastive_loss = contrastive_loss,
                                               writer=writer, global_step=len(val_gt_loader) * e,
                                               name=f"{opt.experiment_dir_name}_by_batch")
        
        print("*"*100)
        print("epoch{}: tacc: {:.4f}, vacc: {:.4f}, tsrcc: {:.4f}, vsrcc: {:.4f}, test_loss: {:.4f}, val_loss: {:.4f}".format(e, tacc, vacc, tsrcc, vsrcc, test_loss, val_loss))
        print("epoch{}: tacc_gt: {:.4f}, vacc_gt: {:.4f}, tsrcc_gt: {:.4f}, vsrcc_gt: {:.4f}, test_loss_gt: {:.4f}, val_loss_gt: {:.4f}".format(e, tacc_gt, vacc_gt, tsrcc_gt, vsrcc_gt, test_loss_gt, val_loss_gt))
        # if e < 20:
        #     scheduler.step()
        # else:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = opt.init_lr * 0.1
        
        scheduler.step()


        # save five checkpoints
        if tacc > best_test_acc:
            if best_test_model_name is not None:
                try:
                    os.remove(os.path.join(opt.model_save_path, best_test_model_name))
                except FileNotFoundError:
                    pass
            best_test_acc = tacc
            best_test_model_name = f"best_test_model_{e}_best_tacc{best_test_acc:.4f}_vacc_{vacc:.4f}_tacc_gt_{tacc_gt:.4f}_vacc_gt_{vacc_gt:.4f}.pth"
            print("**************************************best test model saved for epoch {}**************************************".format(e))
            torch.save({
                'epoch': e + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(opt.model_save_path, best_test_model_name))

        if vacc > best_val_acc:
            if best_val_model_name is not None:
                try:
                    os.remove(os.path.join(opt.model_save_path, best_val_model_name))
                except FileNotFoundError:
                    pass
            best_val_acc = vacc
            best_val_model_name = f"best_val_model_{e}_best_vacc{best_val_acc:.4f}_tacc_{tacc:.4f}_vacc_gt_{vacc_gt:.4f}_tacc_gt_{tacc_gt:.4f}.pth"
            print("**************************************best validation model saved for epoch {}**************************************".format(e))
            torch.save({
                'epoch': e + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(opt.model_save_path, best_val_model_name))
        
        if vacc_gt > best_val_acc_gt:
            if best_val_gt_model_name is not None:
                try:
                    os.remove(os.path.join(opt.model_save_path, best_val_gt_model_name))
                except FileNotFoundError:
                    pass
            best_val_acc_gt = vacc_gt
            best_val_gt_model_name = f"best_val_gt_model_{e}_best_vacc_gt{best_val_acc_gt:.4f}_tacc_gt_{tacc_gt:.4f}_vacc_{vacc:.4f}_tacc_{tacc:.4f}.pth"
            print("**************************************best validation GT model saved for epoch {}**************************************".format(e))
            torch.save({
                'epoch': e + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(opt.model_save_path, best_val_gt_model_name))
        
        if tacc_gt > best_test_acc_gt:
            if best_test_gt_model_name is not None:
                try:
                    os.remove(os.path.join(opt.model_save_path, best_test_gt_model_name))
                except FileNotFoundError:
                    pass
            best_test_acc_gt = tacc_gt
            best_test_gt_model_name = f"best_test_gt_model_{e}_best_tacc_gt{best_test_acc_gt:.4f}_vacc_gt_{vacc_gt:.4f}_tacc_{tacc:.4f}_vacc_{vacc:.4f}.pth"
            print("**************************************best test GT model saved for epoch {}**************************************".format(e))
            torch.save({
                'epoch': e + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(opt.model_save_path, best_test_gt_model_name))

        both_acc = (vacc + tacc) / 2
        if tacc >= best_test_acc and vacc >= best_val_acc and both_acc > best_both_acc:
            if best_both_model_name is not None:
                try:
                    os.remove(os.path.join(opt.model_save_path, best_both_model_name))
                except FileNotFoundError:
                    pass
            best_both_acc = both_acc
            best_both_model_name = f"best_both_model_{e}_best_acc{both_acc:.4f}_tacc_{tacc:.4f}_vacc_{vacc:.4f}_tacc_gt_{tacc_gt:.4f}_vacc_gt_{vacc_gt:.4f}.pth"
            print("**************************************best both model saved for epoch {}**************************************".format(e))
            torch.save({
                'epoch': e + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(opt.model_save_path, best_both_model_name))
        # save log

        f.write('epoch:%d,t_acc:%.5f,v_acc:%.5f,tacc_gt:%.5f, vacc_gt:%.5f,train:%.5f,test:%.5f, contrastive_loss":%.5f\r\n' % (e, tacc, vacc,tacc_gt, vacc_gt, train_loss, test_loss, contra_loss))
        f.flush()
        writer.add_scalars("epoch_loss", {'train': train_loss, 'test': test_loss}, global_step=e)

        writer.add_scalars("acc",{'test_acc': tacc, 'val_acc': vacc}, global_step=e)

    # save last epoch's model
    last_model_name = f"last_epoch_{opt.num_epoch - 1}_tacc_{tacc:.4f}_tsrcc_{tsrcc:.4f}_tplcc_{tplcc:.4f}_vacc_{vacc:.4f}_vsrcc_{vsrcc:.4f}.pth"
    print("**************************************last epoch model saved**************************************")
    torch.save({
        'epoch': opt.num_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(opt.model_save_path, last_model_name))

    writer.close()
    f.close()


if __name__ =="__main__":
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    start_train(opt)
