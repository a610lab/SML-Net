import sys
import torch
from utils import loss_functions
from utils import MT_utils
from tqdm import tqdm
from utils.train_utils import *
import torch.nn.functional as F

gn = MT_utils.GaussianNoise()


def train_one_epoch(model, optimizer, labeled_data_loader, unlabeled_data_loader, device, epoch):
    ce_loss_function = torch.nn.CrossEntropyLoss().to(device)
    total_loss = AverageMeter()
    total_loss_seg = AverageMeter()
    total_loss_cla = AverageMeter()

    loader = zip(labeled_data_loader, unlabeled_data_loader)
    data_loader = tqdm(loader)
    for i, ((img_x, mask_x, labels_l), (img_u, _, labels_u)) in enumerate(data_loader):
        labels = torch.cat((labels_l, labels_u), dim=0)
        img_x, mask_x, img_u = img_x.to(device), mask_x.to(device), img_u.to(device)
        model.train()
        pred = model(torch.cat((img_x, img_u)))
        loss_seg = ce_loss_function(pred[-2][:img_x.shape[0]], mask_x.to(device))
        loss_cla = ce_loss_function(pred[-1], labels.to(device))

        loss = loss_cla+loss_seg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss.update(loss.item())
        total_loss_seg.update(loss_seg.item())
        total_loss_cla.update(loss_cla.item())

        data_loader.desc = "[train epoch {}] Total loss: {:.3f}, Loss seg: {:.3f}, Loss cla: {:.3f}".format(
            epoch + 1,
            total_loss.avg,
            total_loss_seg.avg,
            total_loss_cla.avg)

    return total_loss.avg


def train_one_epoch_segaug(model, optimizer, labeled_data_loader, unlabeled_data_loader, device, epoch):
    ce_loss_function = torch.nn.CrossEntropyLoss().to(device)
    total_loss = AverageMeter()
    total_loss_seg = AverageMeter()
    total_loss_cla = AverageMeter()

    loader = zip(labeled_data_loader, unlabeled_data_loader)
    data_loader = tqdm(loader)
    for i, ((img_x, mask_x, labels_l), (img_u, _, labels_u)) in enumerate(data_loader):
        labels = torch.cat((labels_l, labels_u), dim=0)
        img_x, mask_x, img_u = img_x.to(device), mask_x.to(device), img_u.to(device)
        model.train()
        pred = model(torch.cat((img_x, img_u)))
        loss_seg = ce_loss_function(pred[-2][:img_x.shape[0]], mask_x.to(device))
        loss_cla = ce_loss_function(pred[-1], labels.to(device))

        loss = loss_cla * 0.5 + loss_seg * 0.5

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss.update(loss.item())
        total_loss_seg.update(loss_seg.item())
        total_loss_cla.update(loss_cla.item())

        data_loader.desc = "[train epoch {}] Total loss: {:.3f}, Loss seg: {:.3f}, Loss cla: {:.3f}".format(
            epoch + 1,
            total_loss.avg,
            total_loss_seg.avg,
            total_loss_cla.avg)

    return total_loss.avg


def train_one_epoch_mt(model_s, model_t, optimizer, labeled_data_loader, unlabeled_data_loader, device, epoch):
    mse_loss_function = loss_functions.softmax_consistency_loss
    ce_loss_function = torch.nn.CrossEntropyLoss().to(device)
    total_loss = AverageMeter()
    total_loss_seg = AverageMeter()
    total_loss_cla = AverageMeter()

    loader = zip(labeled_data_loader, unlabeled_data_loader)
    data_loader = tqdm(loader)

    model_s.train()
    model_t.train()

    for i, ((img_x, mask_x, labels_l), (img_u, _, labels_u)) in enumerate(data_loader):
        labels = torch.cat((labels_l, labels_u), dim=0)
        img_x, img_u, mask_x, labels = img_x.to(device), img_u.to(device), mask_x.type(torch.LongTensor).to(device), labels.to(device)

        input_img = torch.cat((img_x, img_u))
        output_s = model_s(input_img)
        output_t = model_t(input_img)

        loss_x = ce_loss_function(output_s[-2][:mask_x.shape[0]], mask_x)
        loss_u = mse_loss_function(output_s[-2][mask_x.shape[0]:], output_t[-2][mask_x.shape[0]:], 'MSE')
        loss_a = mse_loss_function(output_t[-3][mask_x.shape[0]:], output_t[-2][mask_x.shape[0]:], 'MSE')
        loss_seg = loss_x + loss_u + loss_a
        loss_cla = ce_loss_function(output_s[-1], labels.to(device))
        loss = loss_seg + loss_cla

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        MT_utils.update_ema_variables(model_s, model_t, 0.999, epoch)

        total_loss.update(loss.item())
        total_loss_seg.update(loss_seg.item())
        total_loss_cla.update(loss_cla.item())

        data_loader.desc = "[train epoch {}] Total loss: {:.3f}, Loss seg: {:.3f}, Loss cla: {:.3f}".format(
            epoch + 1,
            total_loss.avg,
            total_loss_seg.avg,
            total_loss_cla.avg)

    return total_loss.avg


def train_one_epoch_swm(model, optimizer, labeled_data_loader, unlabeled_data_loader, device, epoch, conf_thresh):
    ce_loss_function = torch.nn.CrossEntropyLoss(reduction='none').to(device)
    swm_loss_function = loss_functions.sample_weight_loss
    total_loss = AverageMeter()
    total_loss_x = AverageMeter()
    total_loss_s = AverageMeter()
    total_loss_w_fp = AverageMeter()
    total_mask_ratio = AverageMeter()

    loader = zip(labeled_data_loader, unlabeled_data_loader, unlabeled_data_loader)
    data_loader = tqdm(loader)
    for i, ((img_x, mask_x, labels_l),
            (img_u_w, img_u_s1, img_u_s2, cutmix_box1, cutmix_box2, labels_u),
            (img_u_w_mix, img_u_s1_mix, img_u_s2_mix, _, _, _)) in enumerate(data_loader):
        labels = torch.cat((labels_l, labels_u), dim=0)
        img_x, mask_x, img_u_w, img_u_s1, img_u_s2 = img_x.to(device), mask_x.to(device), img_u_w.to(
            device), img_u_s1.to(device), img_u_s2.to(device)
        cutmix_box1, cutmix_box2 = cutmix_box1.to(device), cutmix_box2.to(device)
        img_u_w_mix, img_u_s1_mix, img_u_s2_mix = img_u_w_mix.to(device), img_u_s1_mix.to(device), img_u_s2_mix.to(
            device)

        with torch.no_grad():
            model.eval()
            pred_u_w_mix = model(img_u_w_mix)[-2].detach()
            conf_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)[0]
            mask_u_w_mix = pred_u_w_mix.argmax(dim=1)

        img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = \
            img_u_s1_mix[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]
        img_u_s2[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1] = \
            img_u_s2_mix[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1]

        model.train()

        num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]

        (preds, preds_fp), pred_cla = model(torch.cat((img_x, img_u_w)), need_fp=True)
        pred_x, pred_u_w = preds.split([num_lb, num_ulb])
        pred_u_w_fp = preds_fp[num_lb:]

        pred_u_s1, pred_u_s2 = model(torch.cat((img_u_s1, img_u_s2)))[-2].chunk(2)

        sample_weight = (get_sample_weight(pred_cla, labels) * labels.shape[0]).to(device)

        pred_u_w = pred_u_w.detach()
        conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
        mask_u_w = pred_u_w.argmax(dim=1)

        mask_u_w_cutmixed1, conf_u_w_cutmixed1 = \
            mask_u_w.clone(), conf_u_w.clone()
        mask_u_w_cutmixed2, conf_u_w_cutmixed2 = \
            mask_u_w.clone(), conf_u_w.clone()

        mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w_mix[cutmix_box1 == 1]
        conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w_mix[cutmix_box1 == 1]

        mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w_mix[cutmix_box2 == 1]
        conf_u_w_cutmixed2[cutmix_box2 == 1] = conf_u_w_mix[cutmix_box2 == 1]

        loss_x = (ce_loss_function(pred_x, mask_x).mean(dim=1).mean(dim=1) * sample_weight[:pred_x.shape[0]]).mean()

        loss_u_s1 = ce_loss_function(pred_u_s1, mask_u_w_cutmixed1)
        loss_u_s1 = loss_u_s1 * (conf_u_w_cutmixed1 >= conf_thresh)
        loss_u_s1 = (loss_u_s1.mean(dim=1).mean(dim=1) * sample_weight[pred_u_w.shape[0]:]).mean()

        loss_u_s2 = ce_loss_function(pred_u_s2, mask_u_w_cutmixed2)
        loss_u_s2 = loss_u_s2 * (conf_u_w_cutmixed2 >= conf_thresh)
        loss_u_s2 = (loss_u_s2.mean(dim=1).mean(dim=1) * sample_weight[pred_u_w.shape[0]:]).mean()

        loss_u_w_fp = ce_loss_function(pred_u_w_fp, mask_u_w)
        loss_u_w_fp = loss_u_w_fp * (conf_u_w >= conf_thresh)
        loss_u_w_fp = (loss_u_w_fp.mean(dim=1).mean(dim=1) * sample_weight[pred_u_w.shape[0]:]).mean()

        loss_seg = (loss_x + loss_u_s1 * 0.25 + loss_u_s2 * 0.25 + loss_u_w_fp * 0.5) / 2.0
        loss_cla = ce_loss_function(pred_cla, labels.to(device)).mean()
        loss = loss_seg * 0.5 + loss_cla * 0.5
        # loss = loss_x

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss.update(loss.item())
        seg_loss.update(loss.item())
        cla_loss.update(loss.item())
        total_loss_x.update(loss_x.item())
        total_loss_s.update((loss_u_s1.item() + loss_u_s2.item()) / 2.0)
        total_loss_w_fp.update(loss_u_w_fp.item())

        mask_ratio = (conf_u_w >= conf_thresh).sum() / conf_u_w.sum()
        total_mask_ratio.update(mask_ratio.item())

        data_loader.desc = "[train epoch {}] Total loss: {:.3f}, Loss x: {:.3f}, Loss s: {:.3f}, Loss w_fp: {:.3f}, Mask ratio: {:.3f}".format(
            epoch + 1,
            total_loss.avg,
            total_loss_x.avg,
            total_loss_s.avg,
            total_loss_w_fp.avg,
            total_mask_ratio.avg)

    return total_loss.avg


def train_one_epoch_rwm(model_s, model_t, optimizer, labeled_data_loader, unlabeled_data_loader, device, epoch):
    mse_loss_function = loss_functions.softmax_consistency_loss
    swm_loss_function = loss_functions.sample_weight_loss
    ce_loss_function = torch.nn.CrossEntropyLoss().to(device)
    total_loss = AverageMeter()
    total_loss_seg = AverageMeter()
    total_loss_cla = AverageMeter()

    loader = zip(labeled_data_loader, unlabeled_data_loader)
    data_loader = tqdm(loader)

    model_s.train()
    model_t.train()

    for i, ((img_x, mask_x, labels_l), (img_u, _, labels_u)) in enumerate(data_loader):
        labels = torch.cat((labels_l, labels_u), dim=0)
        img_x, mask_x = img_x.cuda(), mask_x.cuda()
        img_u = img_u.to(device)

        input_img = torch.cat((img_x, img_u))
        input_img_s = gn(input_img)
        input_img_t = gn(input_img)

        output_t = model_t(input_img_t)
        output_s = model_s(input_img_s, output_t[0])

        loss_x = ce_loss_function(output_s[-2][:mask_x.shape[0]], mask_x)
        loss_u = mse_loss_function(output_s[-2][mask_x.shape[0]:], output_t[-2][mask_x.shape[0]:], 'MSE')

        loss_seg = loss_x + loss_u
        loss_cla = ce_loss_function(output_s[-1], labels.to(device))
        loss = loss_seg + loss_cla

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        MT_utils.update_ema_variables(model_s, model_t, 0.999, epoch)

        total_loss.update(loss.item())
        total_loss_seg.update(loss_seg.item())
        total_loss_cla.update(loss_cla.item())

        data_loader.desc = "[train epoch {}] Total loss: {:.3f}, Loss seg: {:.3f}, Loss cla: {:.3f}".format(
            epoch + 1,
            total_loss.avg,
            total_loss_seg.avg,
            total_loss_cla.avg)

    return total_loss.avg


def train_one_epoch_fixmatch(model, optimizer, labeled_data_loader, unlabeled_data_loader, device, epoch, conf_thresh):
    ce_loss_function = torch.nn.CrossEntropyLoss(reduction='none').to(device)
    total_loss = AverageMeter()
    total_loss_x = AverageMeter()
    total_loss_s = AverageMeter()
    total_mask_ratio = AverageMeter()

    loader = zip(labeled_data_loader, unlabeled_data_loader, unlabeled_data_loader)
    data_loader = tqdm(loader)
    for i, ((img_x, mask_x, labels_l),
            (img_u_w, img_u_s, _, cutmix_box, _, labels_u),
            (img_u_w_mix, img_u_s_mix, _, _, _, _)) in enumerate(data_loader):
        labels = torch.cat((labels_l, labels_u), dim=0)
        img_x, mask_x, img_u_w, img_u_s = img_x.to(device), mask_x.to(device), img_u_w.to(device), img_u_s.to(device)
        cutmix_box1 = cutmix_box.to(device)
        img_u_w_mix, img_u_s_mix = img_u_w_mix.to(device), img_u_s_mix.to(device)

        with torch.no_grad():
            model.eval()
            pred_u_w_mix = model(img_u_w_mix)[-2].detach()
            conf_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)[0]
            mask_u_w_mix = pred_u_w_mix.argmax(dim=1)

        img_u_s[cutmix_box1.unsqueeze(1).expand(img_u_s.shape) == 1] = \
            img_u_s_mix[cutmix_box1.unsqueeze(1).expand(img_u_s.shape) == 1]

        model.train()

        num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]

        preds, pred_cla = model(torch.cat((img_x, img_u_w, img_u_s)))
        pred_cla = pred_cla[:num_lb + num_ulb]
        pred_x, pred_u_w, pred_u_s = preds.split([num_lb, num_ulb, num_ulb])
        sample_weight = (get_sample_weight(pred_cla, labels) * labels.shape[0]).to(device)

        pred_u_w = pred_u_w.detach()
        conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
        mask_u_w = pred_u_w.argmax(dim=1)

        mask_u_w_cutmixed1, conf_u_w_cutmixed1 = mask_u_w.clone(), conf_u_w.clone()

        mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w_mix[cutmix_box1 == 1]
        conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w_mix[cutmix_box1 == 1]

        loss_x = (ce_loss_function(pred_x, mask_x).mean(dim=1).mean(dim=1) * sample_weight[:pred_x.shape[0]]).mean()

        loss_u_s1 = ce_loss_function(pred_u_s, mask_u_w_cutmixed1)
        loss_u_s1 = loss_u_s1 * (conf_u_w_cutmixed1 >= conf_thresh)
        loss_u_s1 = (loss_u_s1.mean(dim=1).mean(dim=1) * sample_weight[pred_u_w.shape[0]:]).mean()

        loss_seg = (loss_x + loss_u_s1) / 2.0
        loss_cla = ce_loss_function(pred_cla, labels.to(device)).mean()
        loss = loss_seg * 0.5 + loss_cla * 0.5
        # loss = loss_x

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss.update(loss.item())
        total_loss_x.update(loss_x.item())

        mask_ratio = (conf_u_w >= conf_thresh).sum() / conf_u_w.sum()
        total_mask_ratio.update(mask_ratio.item())

        data_loader.desc = "[train epoch {}] Total loss: {:.3f}, Loss x: {:.3f}, Loss s: {:.3f}, Mask ratio: {:.3f}".format(
            epoch + 1,
            total_loss.avg,
            total_loss_x.avg,
            total_loss_s.avg,
            total_mask_ratio.avg)

    return total_loss.avg


def train_one_epoch_unimatch(model, optimizer, labeled_data_loader, unlabeled_data_loader, device, epoch, conf_thresh):
    ce_loss_function = torch.nn.CrossEntropyLoss().to(device)
    total_loss = AverageMeter()
    total_loss_x = AverageMeter()
    total_loss_s = AverageMeter()
    total_loss_w_fp = AverageMeter()
    total_mask_ratio = AverageMeter()

    loader = zip(labeled_data_loader, unlabeled_data_loader, unlabeled_data_loader)
    data_loader = tqdm(loader)
    for i, ((img_x, mask_x, labels_l),
            (img_u_w, img_u_s1, img_u_s2, cutmix_box1, cutmix_box2, labels_u),
            (img_u_w_mix, img_u_s1_mix, img_u_s2_mix, _, _, _)) in enumerate(data_loader):
        labels = torch.cat((labels_l, labels_u), dim=0)
        img_x, mask_x, img_u_w, img_u_s1, img_u_s2 = img_x.to(device), mask_x.to(device), img_u_w.to(
            device), img_u_s1.to(device), img_u_s2.to(device)
        cutmix_box1, cutmix_box2 = cutmix_box1.to(device), cutmix_box2.to(device)
        img_u_w_mix, img_u_s1_mix, img_u_s2_mix = img_u_w_mix.to(device), img_u_s1_mix.to(device), img_u_s2_mix.to(
            device)

        with torch.no_grad():
            model.eval()
            pred_u_w_mix = model(img_u_w_mix)[-2].detach()
            conf_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)[0]
            mask_u_w_mix = pred_u_w_mix.argmax(dim=1)

        img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = \
            img_u_s1_mix[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]
        img_u_s2[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1] = \
            img_u_s2_mix[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1]

        model.train()

        num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]

        (preds, preds_fp), pred_cla = model(torch.cat((img_x, img_u_w)), need_fp=True)
        pred_x, pred_u_w = preds.split([num_lb, num_ulb])
        pred_u_w_fp = preds_fp[num_lb:]

        pred_u_s1, pred_u_s2 = model(torch.cat((img_u_s1, img_u_s2)))[-2].chunk(2)

        pred_u_w = pred_u_w.detach()
        conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
        mask_u_w = pred_u_w.argmax(dim=1)

        mask_u_w_cutmixed1, conf_u_w_cutmixed1 = \
            mask_u_w.clone(), conf_u_w.clone()
        mask_u_w_cutmixed2, conf_u_w_cutmixed2 = \
            mask_u_w.clone(), conf_u_w.clone()

        mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w_mix[cutmix_box1 == 1]
        conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w_mix[cutmix_box1 == 1]

        mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w_mix[cutmix_box2 == 1]
        conf_u_w_cutmixed2[cutmix_box2 == 1] = conf_u_w_mix[cutmix_box2 == 1]

        loss_x = ce_loss_function(pred_x, mask_x)

        loss_u_s1 = ce_loss_function(pred_u_s1, mask_u_w_cutmixed1)
        loss_u_s1 = loss_u_s1 * (conf_u_w_cutmixed1 >= conf_thresh)
        loss_u_s1 = loss_u_s1.mean()

        loss_u_s2 = ce_loss_function(pred_u_s2, mask_u_w_cutmixed2)
        loss_u_s2 = loss_u_s2 * (conf_u_w_cutmixed2 >= conf_thresh)
        loss_u_s2 = loss_u_s2.mean()

        loss_u_w_fp = ce_loss_function(pred_u_w_fp, mask_u_w)
        loss_u_w_fp = loss_u_w_fp * (conf_u_w >= conf_thresh)
        loss_u_w_fp = loss_u_w_fp.mean()

        loss_seg = (loss_x + loss_u_s1 * 0.25 + loss_u_s2 * 0.25 + loss_u_w_fp * 0.5) / 2.0
        loss_cla = ce_loss_function(pred_cla, labels.to(device)).mean()
        loss = loss_seg * 0.5 + loss_cla * 0.5

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss.update(loss.item())
        total_loss_x.update(loss_x.item())
        total_loss_s.update((loss_u_s1.item() + loss_u_s2.item()) / 2.0)
        total_loss_w_fp.update(loss_u_w_fp.item())

        mask_ratio = (conf_u_w >= conf_thresh).sum() / conf_u_w.sum()
        total_mask_ratio.update(mask_ratio.item())

        data_loader.desc = "[train epoch {}] Total loss: {:.3f}, Loss x: {:.3f}, Loss s: {:.3f}, Loss w_fp: {:.3f}, Mask ratio: {:.3f}".format(
            epoch + 1,
            total_loss.avg,
            total_loss_x.avg,
            total_loss_s.avg,
            total_loss_w_fp.avg,
            total_mask_ratio.avg)

    return total_loss.avg


def train_one_epoch_unimatch_free(model, optimizer, labeled_data_loader, unlabeled_data_loader, device, epoch,
                                  conf_thresh):
    ce_loss_function = torch.nn.CrossEntropyLoss(reduction='none').to(device)
    total_loss = AverageMeter()
    total_loss_cla = AverageMeter()
    total_loss_x = AverageMeter()
    total_loss_s = AverageMeter()
    total_loss_w_fp = AverageMeter()
    total_mask_ratio = AverageMeter()

    loader = zip(labeled_data_loader, unlabeled_data_loader, unlabeled_data_loader)
    data_loader = tqdm(loader)
    for i, ((img_x, mask_x, labels_l),
            (img_u_w, img_u_s1, img_u_s2, cutmix_box1, cutmix_box2, labels_u),
            (img_u_w_mix, img_u_s1_mix, img_u_s2_mix, _, _, _)) in enumerate(data_loader):
        labels = torch.cat((labels_l, labels_u), dim=0)
        img_x, mask_x = img_x.cuda(), mask_x.cuda()
        img_u_w = img_u_w.cuda()
        img_u_s1, img_u_s2 = img_u_s1.cuda(), img_u_s2.cuda()
        cutmix_box1, cutmix_box2 = cutmix_box1.cuda(), cutmix_box2.cuda()
        img_u_w_mix = img_u_w_mix.cuda()
        img_u_s1_mix, img_u_s2_mix = img_u_s1_mix.cuda(), img_u_s2_mix.cuda()

        with torch.no_grad():
            model.eval()
            pred_u_w_mix = model(img_u_w_mix)[-2].detach()
            conf_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)[0]
            mask_u_w_mix = pred_u_w_mix.argmax(dim=1)

        img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = \
            img_u_s1_mix[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]
        img_u_s2[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1] = \
            img_u_s2_mix[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1]

        model.train()

        num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]

        (preds, preds_fp), pred_cla = model(torch.cat((img_x, img_u_w)), need_fp=True)
        pred_x, pred_u_w = preds.split([num_lb, num_ulb])
        pred_u_w_fp = preds_fp[num_lb:]
        sample_weight_x = (get_sample_weight(pred_cla[num_lb:], labels[num_lb:]) * labels_l.shape[0]).to(device)
        sample_weight_u = (1 / (get_sample_weight(pred_cla[:num_lb], labels[:num_lb]) * labels_u.shape[0])).to(device)

        pred_u_s1, pred_u_s2 = model(torch.cat((img_u_s1, img_u_s2)))[-2].chunk(2)

        pred_u_w = pred_u_w.detach()
        conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
        mask_u_w = pred_u_w.argmax(dim=1)

        mask_u_w_cutmixed1, conf_u_w_cutmixed1 = \
            mask_u_w.clone(), conf_u_w.clone()
        mask_u_w_cutmixed2, conf_u_w_cutmixed2 = \
            mask_u_w.clone(), conf_u_w.clone()

        mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w_mix[cutmix_box1 == 1]
        conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w_mix[cutmix_box1 == 1]

        mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w_mix[cutmix_box2 == 1]
        conf_u_w_cutmixed2[cutmix_box2 == 1] = conf_u_w_mix[cutmix_box2 == 1]

        loss_x = (ce_loss_function(pred_x, mask_x).mean(dim=1).mean(dim=1) * sample_weight_x).mean()

        loss_u_s1 = ce_loss_function(pred_u_s1, mask_u_w_cutmixed1)
        loss_u_s1 = loss_u_s1 * (conf_u_w_cutmixed1 >= conf_thresh)
        loss_u_s1 = (loss_u_s1.mean(dim=1).mean(dim=1) * sample_weight_u).mean()

        loss_u_s2 = ce_loss_function(pred_u_s2, mask_u_w_cutmixed2)
        loss_u_s2 = loss_u_s2 * (conf_u_w_cutmixed2 >= conf_thresh)
        loss_u_s2 = (loss_u_s2.mean(dim=1).mean(dim=1) * sample_weight_u).mean()

        loss_u_w_fp = ce_loss_function(pred_u_w_fp, mask_u_w)
        loss_u_w_fp = loss_u_w_fp * (conf_u_w >= conf_thresh)
        loss_u_w_fp = (loss_u_w_fp.mean(dim=1).mean(dim=1) * sample_weight_u).mean()

        loss_u = loss_u_s1 * 0.25 + loss_u_s2 * 0.25 + loss_u_w_fp * 0.5

        loss_seg = loss_x * 0.5 + loss_u * 0.5
        loss_cla = ce_loss_function(pred_cla, labels.to(device)).mean()
        loss = loss_seg + loss_cla * 0.5

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss.update(loss.item())
        total_loss_cla.update(loss_cla.item())
        total_loss_x.update(loss_x.item())
        total_loss_s.update((loss_u_s1.item() + loss_u_s2.item()) / 2.0)
        total_loss_w_fp.update(loss_u_w_fp.item())
        mask_ratio = (conf_u_w >= conf_thresh).sum() / (conf_u_w.shape[0] * conf_u_w.shape[1] * conf_u_w.shape[2])
        total_mask_ratio.update(mask_ratio.item())
        # conf_thresh = 0.99 * conf_thresh + (1 - 0.99) * conf_u_w.max()
        # updata_thresh(cat_thresh, labels, conf_u_w)

        data_loader.desc = "[train epoch {}] Total loss: {:.3f}, Loss cla: {:.3f}, Loss x: {:.3f}, Loss s: {:.3f}, Loss w_fp: {:.3f}, Mask ratio: {:.3f}".format(
            epoch + 1,
            total_loss.avg,
            total_loss_cla.avg,
            total_loss_x.avg,
            total_loss_s.avg,
            total_loss_w_fp.avg,
            total_mask_ratio.avg)

    return total_loss.avg, conf_thresh


def train_one_epoch_our(model_s, model_t, optimizer, labeled_data_loader, unlabeled_data_loader, device, epoch):
    model_s.train()
    model_t.train()
    ce_loss_function = torch.nn.CrossEntropyLoss().to(device)
    sample_weight_loss = loss_functions.sample_weight_loss
    accu_cla_loss = torch.zeros(1).to(device)  # 累计分类损失
    accu_seg_loss = torch.zeros(1).to(device)  # 累计分割损失
    accu_cla_num = torch.zeros(1).to(device)  # 累计分类预测正确的样本数
    accu_seg_num = 0  # 累计分割预测正确的样本数
    optimizer.zero_grad()

    sample_cla_num = 0
    data_loader = tqdm(unlabeled_data_loader)
    labeled_iter = iter(labeled_data_loader)
    for step, data in enumerate(data_loader):
        images_u, _, labels_u = data
        try:
            images_l, mask, labels_l = next(labeled_iter)
        except StopIteration:
            labeled_iter = iter(labeled_data_loader)
            images_l, mask, labels_l = next(labeled_iter)

        images = torch.cat((images_l, images_u), dim=0)
        labels = torch.cat((labels_l, labels_u), dim=0)
        images = gn(images.to(device))
        images_t = gn(images.to(device))

        sample_cla_num += images.shape[0]

        pred_t = model_t(images_t.to(device))
        pred_s = model_s(images.to(device), pred_t[-2])

        pred_s_classes = torch.max(pred_t[-1], dim=1)[1]
        accu_cla_num += torch.eq(pred_s_classes, labels.to(device)).sum()
        accu_seg_num += metric_dsc(pred_t[-2][:labels_l.shape[0]], mask.type(torch.LongTensor).to(device))

        sample_weight = get_sample_weight(pred_s[-1], labels) * labels.shape[0]

        loss_classification = ce_loss_function(pred_s[-1], labels.type(torch.LongTensor).to(device))
        loss_segmentation = sample_weight_loss(pred_s[-2][labels_l.shape[0]:], pred_t[-2][labels_l.shape[0]:],
                                               pred_s[-2][:labels_l.shape[0]], mask.type(torch.LongTensor).to(device),
                                               sample_weight)
        loss = loss_classification + loss_segmentation

        loss.backward()
        accu_cla_loss += loss_classification.detach()
        accu_seg_loss += loss_segmentation.detach()

        data_loader.desc = "[train epoch {}] cla_loss: {:.3f}, seg_loss: {:.3f}, acc: {:.3f}, dsc: {:.3f}".format(
            epoch + 1,
            accu_cla_loss.item() / (step + 1),
            accu_seg_loss.item() / (step + 1),
            accu_cla_num.item() / sample_cla_num,
            accu_seg_num / (step + 1))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()
        MT_utils.update_ema_variables(model_s, model_t, 0.999, epoch)

    accu_loss = accu_cla_loss + accu_seg_loss
    return accu_loss.item() / (step + 1), accu_cla_loss.item() / sample_cla_num, accu_seg_loss.item() / (step + 1)


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    model.eval()

    accu_cla_num = torch.zeros(1).to(device)  # 累计分类预测正确的样本数
    accu_seg_num = torch.zeros(1).to(device)  # 累计分割预测正确的样本数

    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, masks, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred[-1], dim=1)[1]
        accu_cla_num += torch.eq(pred_classes, labels.to(device)).sum()
        accu_seg_num += metric_dsc(pred[-2], masks.type(torch.LongTensor).to(device))

        data_loader.desc = "[valid epoch {}] acc: {:.3f}, dsc: {:.3f}".format(
            epoch + 1,
            accu_cla_num.item() / sample_num,
            accu_seg_num.item() / (step + 1)
        )

    return accu_cla_num.item() / sample_num, accu_seg_num.item() / (step + 1)
