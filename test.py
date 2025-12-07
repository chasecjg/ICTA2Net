import os
import sys
import csv
import argparse

from os import path
import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score

# add parent path
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(os.path.split(rootPath)[0])

from os import path
d = path.dirname(__file__)
parent_path = os.path.dirname(d)
sys.path.append(parent_path)

# import necessary modules
from ICTA2Net.DETRIS.model import build_segmenter
from ICTA2Net.dataset.dataset_test import AVADataset_test
from ICTA2Net.utils.utils import AverageMeter
from ICTA2Net.utils import option

# initialize file for results
# f = open('ICTA2Net/results/FiveK.txt', 'w')

# initialize options
opt = option.init()
opt.device = torch.device("cuda:{}".format(opt.gpu_id))

# create data loader
def create_data_part(opt, csv_path=None, image_path=None):
    csv_path = csv_path
    image_path = image_path
    test_csv_path = os.path.join(opt.path_to_save_csv, csv_path)
    test_ds = AVADataset_test(test_csv_path, root_dir=image_path, if_train=False, ablate_text=opt.ablate_text)
    test_loader = DataLoader(test_ds, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=False)
    return test_loader
# calculate ACC
def validate(opt, model, loader, mse_criterion, writer=None, global_step=None, name=None, csv_save_path=None):
    model.eval()
    validate_losses = AverageMeter()
    true_labels = []
    pred_labels = []
    confidence_list = []

    # open CSV file to save results
    csv_file = open(csv_save_path, 'w', newline='', encoding='utf-8-sig')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['img1_name', 'score1', 'pred1', 'img2_name', 'score2', 'pred2', 'pred_label', 'gt_label'])

    for idx, (img1, img2, text1, text2, score1, score2, labels, image_name1, image_name2, confidence) in enumerate(tqdm(loader)):
        img1, img2 = img1.to(opt.device), img2.to(opt.device)
        text1 = text1.to(opt.device)
        confidence = confidence.to(opt.device).float().view(-1,1)

        with torch.no_grad():
            pred1, pred2,_,_,_,_ = model(img1, img2, text1)


            scores = torch.cat((pred1, pred2), dim=1)  # shape: (B, 2)
            prob_pred1 = torch.softmax(scores * 5, dim=1)[:, 0:1]  

            pred_scores = torch.where(prob_pred1 > 0.5, torch.ones_like(prob_pred1),
                            torch.where(prob_pred1 < 0.5, -torch.ones_like(prob_pred1), torch.zeros_like(prob_pred1)))

            true_scores = labels.view(-1,1).to(opt.device)

            score1 = score1.float().view(-1,1)
            score2 = score2.float().view(-1,1)
            
            pred = 0.5 * (1 + pred_scores)
            labels = 0.5 * (1 + true_scores)

            mse_loss = mse_criterion(pred, labels.float())
            loss = mse_loss
            validate_losses.update(loss.item(), img1.size(0))

            true_labels.extend(true_scores.cpu().numpy().flatten())
            pred_labels.extend(pred_scores.cpu().numpy().flatten())
            confidence_list.extend(confidence.cpu().numpy().flatten())

            # write to csv
            for name1, score1_val, p1_val, name2, score2_val, p2_val, pred_label, gt_label in zip(
                image_name1, score1.cpu().numpy(), pred1.cpu().numpy(),
                image_name2, score2.cpu().numpy(), pred2.cpu().numpy(),
                pred_scores.cpu().numpy(), true_scores.cpu().numpy()):
                csv_writer.writerow([
                    name1, score1_val[0], p1_val[0],
                    name2, score2_val[0], p2_val[0],
                    pred_label[0], gt_label[0]
                ])

        if writer is not None:
            writer.add_scalar(f"{name}/val_loss.avg", validate_losses.avg, global_step=global_step + idx)

    csv_file.close()

    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    confidence_list = np.array(confidence_list).flatten()

    srcc, _ = spearmanr(true_labels, pred_labels)
    plcc, _ = pearsonr(true_labels, pred_labels)
    accuracy = accuracy_score(true_labels, pred_labels)

    print(f"Weighted Accuracy: {accuracy:.4f}, SRCC: {srcc:.4f}, PLCC: {plcc:.4f}")
    return validate_losses.avg, accuracy, srcc, plcc



# calculate IEA 
def validate_iea(opt, model, loader, mse_criterion, writer=None, global_step=None, name=None, csv_save_path=None):
    model.eval()
    validate_losses = AverageMeter()
    true_labels = []
    pred_labels = []
    confidence_list = []

    # open CSV file to save data
    csv_file = open(csv_save_path, 'w', newline='', encoding='utf-8-sig')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['img1_name', 'score1', 'pred1', 'img2_name', 'score2', 'pred2', 'pred_label', 'gt_label'])

    for idx, (img1, img2, text1, text2, score1, score2, labels, image_name1, image_name2, confidence) in enumerate(tqdm(loader)):
        img1, img2 = img1.to(opt.device), img2.to(opt.device)
        text1 = text1.to(opt.device)
        confidence = confidence.to(opt.device).float().view(-1,1)

        with torch.no_grad():
            pred1, pred2,_,_,_,_  = model(img1, img2, text1)

            scores = torch.cat((pred1, pred2), dim=1)  # shape: (B, 2)
            prob_pred1 = torch.softmax(scores * 5, dim=1)[:, 0:1]  

            pred_scores = torch.where(prob_pred1 > 0.5, torch.ones_like(prob_pred1),
                            torch.where(prob_pred1 < 0.5, -torch.ones_like(prob_pred1), torch.zeros_like(prob_pred1)))

            true_scores = labels.view(-1,1).to(opt.device)

            score1 = score1.float().view(-1,1)
            score2 = score2.float().view(-1,1)
            
            pred = 0.5 * (1 + pred_scores)
            labels = 0.5 * (1 + true_scores)

            mse_loss = mse_criterion(pred, labels.float())
            loss = mse_loss
            validate_losses.update(loss.item(), img1.size(0))

            true_labels.extend(true_scores.cpu().numpy().flatten())
            pred_labels.extend(pred_scores.cpu().numpy().flatten())
            confidence_list.extend(confidence.cpu().numpy().flatten())

            # write to csv
            for name1, score1_val, p1_val, name2, score2_val, p2_val, pred_label, gt_label in zip(
                image_name1, score1.cpu().numpy(), pred1.cpu().numpy(),
                image_name2, score2.cpu().numpy(), pred2.cpu().numpy(),
                pred_scores.cpu().numpy(), true_scores.cpu().numpy()):
                csv_writer.writerow([
                    name1, score1_val[0], p1_val[0],
                    name2, score2_val[0], p2_val[0],
                    pred_label[0], gt_label[0]
                ])

        if writer is not None:
            writer.add_scalar(f"{name}/val_loss.avg", validate_losses.avg, global_step=global_step + idx)

    csv_file.close()

    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    confidence_list = np.array(confidence_list).flatten()

    srcc, _ = spearmanr(true_labels, pred_labels)
    plcc, _ = pearsonr(true_labels, pred_labels)

    correct = (true_labels == pred_labels).astype(np.float32)
    weighted_correct = (correct * confidence_list).sum()
    total_confidence = confidence_list.sum()
    weighted_accuracy = weighted_correct / total_confidence if total_confidence > 0 else 0.0
    print(f"Weighted Accuracy: {weighted_accuracy:.4f}, SRCC: {srcc:.4f}, PLCC: {plcc:.4f}")
    return validate_losses.avg, weighted_accuracy, srcc, plcc
# test
def start_test(opt, csv_path, image_path):
    test_loader = create_data_part(opt, csv_path, image_path)
    model, _ = build_segmenter(opt)
    criterion = nn.MSELoss()

    model = model.to(opt.device)
    criterion.to(opt.device)

    if opt.resume:
        checkpoint = torch.load(opt.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("************ Model Loaded Successfully ************")

    print(f"********************* Start Testing {csv_path} *********************")
    base_name = os.path.splitext(csv_path)[0]
    csv_save_path =f"ICTA2Net/results/{base_name}_test.csv"
    result_txt_path = f"ICTA2Net/results/{base_name}_test.txt"
    test_loss, tacc, srcc, plcc = validate(opt, model=model, loader=test_loader, mse_criterion=criterion, csv_save_path=csv_save_path)
    test_loss_iea, tacc_iea, srcc_iea, plcc_iea = validate_iea(opt, model=model, loader=test_loader, mse_criterion=criterion, csv_save_path=csv_save_path)

    with open(result_txt_path, 'w') as f:
        f.write(f"test_loss: {test_loss}\n")
        f.write(f"tacc: {tacc}\n")
        f.write(f"srcc: {srcc}\n")
        f.write(f"plcc: {plcc}\n")
        f.write(f"test_loss_iea: {test_loss_iea}\n")
        f.write(f"tacc_iea: {tacc_iea}\n")
        f.write(f"srcc_iea: {srcc_iea}\n")
        f.write(f"plcc_iea: {plcc_iea}\n")
    print(f"********************* Finished Testing {csv_path} *********************")

if __name__ == "__main__":
    path_map = {
        'ICTAA-GP.csv': opt.image_path_G,
        'ICTAA-HP.csv': opt.image_path_H,
        'ICTAA-GF.csv': opt.image_path_G,
        'ICTAA-HF.csv': opt.image_path_H,
    }
    # you can test your datasets by uncommenting and modifying the path_map like following:
    # path_map = {
    #     'realsee_test_delete_demo_pinehole_v3.csv': opt.image_path_realsee
    # }   
    for csv_name, image_path in path_map.items():
        start_test(opt, csv_name, image_path)
