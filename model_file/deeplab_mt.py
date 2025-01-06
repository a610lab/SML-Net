import timm

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd.variable import Variable
import torchvision.ops as ops
from model_file.backbone import densenet, Resnest
from utils.triple import TripletModel


class DeepLabV3Plus(nn.Module):
    def __init__(self, model_name, num_class_seg=2, num_class_cla=3):
        super(DeepLabV3Plus, self).__init__()

        if model_name in 'densenet':
            self.backbone = densenet.densenet121()

        low_channels = 256
        high_channels = 1024
        self.gn = GaussianNoise()

        self.head = ASPPModule(high_channels, [6, 12, 18])

        self.reduce = nn.Sequential(nn.Conv2d(low_channels, 48, 1, bias=False),
                                    nn.BatchNorm2d(48),
                                    nn.ReLU(True))

        self.fuse = nn.Sequential(nn.Conv2d(high_channels // 8 + 48, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True),
                                  nn.Conv2d(256, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True))
        self.head1 = ASPPModule1(1024, [6, 12, 18])
        self.fuse1 = nn.Sequential(nn.Conv2d(low_channels, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True),
                                  nn.Conv2d(256, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True))

        self.classifier = nn.Conv2d(256, num_class_seg, 1, bias=True)

        self.tr = TripletModel()

        self.gau1 = GAU(1024, 1024)
        self.gau2 = GAU(1024, 512)
        self.gau3 = GAU(512, 256)

    def forward(self, x, seg_t=None):
        h, w = x.shape[-2:]

        x = self.gn(x)
        feats = self.backbone.forward_features(x)
        c1, c2, c3, c4 = feats[0], feats[1], feats[2], feats[-1]
        # print(c1.shape, c2.shape, c3.shape, c4.shape) #torch.Size([16, 256, 24, 32]) torch.Size([16, 512, 12, 16]) torch.Size([16, 1024, 6, 8]) torch.Size([16, 1024, 3, 4])

        out = self._decode(c1, c4)
        out_s = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)

        part = combined_region_weight_module(c4, out_s, seg_t)

        temp = self.tr(c4)

        x = c4 + part + temp

        out_c = self.backbone.forward_head(x)

        c1 = region_location_module(c1, out_s)
        c2 = region_location_module(c2, out_s)
        c3 = region_location_module(c3, out_s)
        c4 = region_location_module(c4, out_s)

        am = self.gau1(c4, c3)
        am = self.gau2(am, c2)
        am = self.gau3(am, c1)
        am = self.fuse1(am)
        out1 = self.classifier(am)
        out1 = F.interpolate(out1, size=(h, w), mode="bilinear", align_corners=True)

        return out1, out_s, out_c

    def _decode(self, c1, c4):
        c4 = self.head(c4)
        c4 = F.interpolate(c4, size=c1.shape[-2:], mode="bilinear", align_corners=True)
        c1 = self.reduce(c1)
        feature = torch.cat([c1, c4], dim=1)
        feature = self.fuse(feature)
        out = self.classifier(feature)

        return out

def ASPPConv(in_channels, out_channels, atrous_rate):
    block = nn.Sequential(
                          nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate,
                                    dilation=atrous_rate, bias=False),
                          nn.BatchNorm2d(out_channels),
                          nn.ReLU(True))
    return block


class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__()
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU(True))

    def forward(self, x):
        h, w = x.shape[-2:]
        pool = self.gap(x)
        return F.interpolate(pool, (h, w), mode="bilinear", align_corners=True)


class ASPPModule(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPPModule, self).__init__()
        out_channels = in_channels // 8
        rate1, rate2, rate3 = atrous_rates

        self.b0 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                nn.BatchNorm2d(out_channels),
                                nn.ReLU(True))
        self.b1 = ASPPConv(in_channels, out_channels, rate1)
        self.b2 = ASPPConv(in_channels, out_channels, rate2)
        self.b3 = ASPPConv(in_channels, out_channels, rate3)
        self.b4 = ASPPPooling(in_channels, out_channels)

        self.project = nn.Sequential(nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
                                     nn.BatchNorm2d(out_channels),
                                     nn.ReLU(True))

    def forward(self, x):
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = self.b4(x)
        y = torch.cat((feat0, feat1, feat2, feat3, feat4), 1)
        return self.project(y)


class ASPPModule1(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPPModule1, self).__init__()
        out_channels = in_channels
        rate1, rate2, rate3 = atrous_rates

        self.b0 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                nn.BatchNorm2d(out_channels),
                                nn.ReLU(True))
        self.b1 = ASPPConv(in_channels, out_channels, rate1)
        self.b2 = ASPPConv(in_channels, out_channels, rate2)
        self.b3 = ASPPConv(in_channels, out_channels, rate3)
        self.b4 = ASPPPooling(in_channels, out_channels)

        self.project = nn.Sequential(nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
                                     nn.BatchNorm2d(out_channels),
                                     nn.ReLU(True))

    def forward(self, x):
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = self.b4(x)
        y = torch.cat((feat0, feat1, feat2, feat3, feat4), 1)
        return self.project(y)


class GaussianNoise(nn.Module):
    def __init__(self, std=0.05):
        super(GaussianNoise, self).__init__()
        self.std1 = std

    def forward(self, x):
        noise1 = Variable(torch.zeros(x.shape).cuda())
        self.register_buffer('noise2', noise1)  # My own contribution , registering buffer for data parallel usage
        c = x.shape[0]
        self.noise2.data.normal_(0, std=self.std1)
        x = x + self.noise2[:c]
        return x


def region_weight_module(x, seg_s, seg_t=None, mask_background=False):
    def get_background_masked_image(seg):
        background_masked_image = torch.zeros_like(seg)
        background_masked_image[:, 0, :, :] = seg[:, 0, :, :]
        return background_masked_image

    threshold = 0.5
    sea = SEAttention(2).to('cuda:0')

    if mask_background:
        seg_s = get_background_masked_image(seg_s)

    weight_map = F.interpolate(seg_s, size=(x.shape[-2], x.shape[-1]), mode='nearest')  # torch.Size([16, 2, 3, 4])
    weight_map = sea(weight_map)
    weight_map[weight_map <= threshold] = 0

    if seg_t is not None:
        if mask_background:
            seg_t = get_background_masked_image(seg_t)
        weight_map_t = F.interpolate(seg_t, size=(x.shape[-2], x.shape[-1]), mode='nearest')
        weight_map_t = sea(weight_map_t)
        weight_map_t[weight_map_t <= threshold] = 0
        weight_confidence = get_dice_weight(seg_s, seg_t)
        weight_confidence = weight_confidence[:, None, None, None]
        weight_map = (weight_map_t + weight_map) / 2 * weight_confidence

    softmax_weight_map = torch.softmax(weight_map, 1)
    softmax_weight_map = softmax_weight_map[:, 1, :, :]
    softmax_weight_map = torch.unsqueeze(softmax_weight_map, dim=1)
    part = torch.mul(x, softmax_weight_map)

    return 1 - part if mask_background else part


def combined_region_weight_module(x, seg_s, seg_t=None):
    part = region_weight_module(x, seg_s, seg_t, mask_background=False)
    back = region_weight_module(x, seg_s, seg_t, mask_background=True)

    return x + part + back


def get_dice_weight(logits, targets):
    smooth = 1
    score_sum = []
    for i in range(logits.size(0)):
        logit = logits[i, 1:, :, :]
        target = targets[i, 1:, :, :]
        bs = target.size(0)
        m1 = torch.sigmoid(logit)
        m2 = torch.sigmoid(target)
        m1 = m1.view(bs, -1)
        m2 = m2.view(bs, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = score.sum() / bs
        score_sum.append(score)
    score_sum = torch.tensor(score_sum)
    return score_sum.cuda()




class SEAttention(nn.Module,):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class GAU(nn.Module):
    def __init__(self, channels_high, channels_low, upsample=True):
        super(GAU, self).__init__()
        # Global Attention Upsample
        self.upsample = upsample
        self.conv3x3 = nn.Conv2d(channels_low, channels_low, kernel_size=3, padding=1, bias=False)
        self.bn_low = nn.BatchNorm2d(channels_low)

        self.conv1x1 = nn.Conv2d(channels_high, channels_low, kernel_size=1, padding=0, bias=False)
        self.bn_high = nn.BatchNorm2d(channels_low)

        if upsample:
            self.conv_upsample = nn.ConvTranspose2d(channels_high, channels_low, kernel_size=4, stride=2, padding=1, bias=False)
            self.bn_upsample = nn.BatchNorm2d(channels_low)
        else:
            self.conv_reduction = nn.Conv2d(channels_high, channels_low, kernel_size=1, padding=0, bias=False)
            self.bn_reduction = nn.BatchNorm2d(channels_low)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, fms_high, fms_low, fm_mask=None):

        b, c, h, w = fms_high.shape

        fms_high_gp = nn.AvgPool2d(fms_high.shape[2:])(fms_high).view(len(fms_high), c, 1, 1)
        fms_high_gp = self.conv1x1(fms_high_gp)
        fms_high_gp = self.bn_high(fms_high_gp)
        fms_high_gp = self.relu(fms_high_gp)


        fms_low_mask = self.conv3x3(fms_low)
        fms_low_mask = self.bn_low(fms_low_mask)

        fms_att = fms_low_mask * fms_high_gp

        if self.upsample:
            out = self.relu(
                self.bn_upsample(self.conv_upsample(fms_high)) + fms_att)
        else:
            out = self.relu(
                self.bn_reduction(self.conv_reduction(fms_high)) + fms_att)

        return out


def region_location_module(x, seg):
    base_ratio = x.shape[-1] / 7
    roi_mask = torch.softmax(seg, dim=1)[:, 1, :, :]
    roi_mask = roi_mask > 0.5
    roi_mask = roi_mask.unsqueeze(1).float()  # Add an extra dimension for broadcasting

    # Initialize output tensor with zeros to avoid repeated allocation
    operated_x = torch.zeros_like(x).cuda()

    for index in range(x.shape[0]):
        if roi_mask[index].sum() == 0:
            operated_x[index] = x[index]
            continue

            # Find bounding box of the ROI
        y_inds, x_inds = torch.where(roi_mask[index, 0])
        y1, x1 = y_inds.min(), x_inds.min()
        y2, x2 = y_inds.max(), x_inds.max()

        # Adjust bounding box based on base_ratio
        new_y1 = max(0, y1 - int(base_ratio))
        new_x1 = max(0, x1 - int(base_ratio))
        new_y2 = min(x.shape[-2], y2 + int(base_ratio))
        new_x2 = min(x.shape[-1], x2 + int(base_ratio))

        # region
        if new_y2 >= new_y1 and new_x2 >= new_x1:
            roi = x[index, :, new_y1:new_y2 + 1, new_x1:new_x2 + 1]
            if roi.shape[-2] > 0 and roi.shape[-1] > 0:
                roi = F.interpolate(roi.unsqueeze(0), size=x.shape[-2:], mode='nearest')
                operated_x[index] = roi.squeeze(0)
            else:
                operated_x[index] = x[index]
        else:
            operated_x[index] = x[index]

            # Apply enhancement only to the cropped region
        enhancement_factor = 1
        operated_x[index, :, new_y1:new_y2 + 1, new_x1:new_x2 + 1] *= (1 + enhancement_factor)

    return operated_x
