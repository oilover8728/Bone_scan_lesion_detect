import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import math

import torchvision
import torchvision.models.detection._utils as det_utils
import torchvision.ops.boxes as box_ops
from torchvision.models.detection import *
from torchvision.models import ResNet
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor, _validate_trainable_layers, BackboneWithFPN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.fcos import FCOS
from torch.hub import load_state_dict_from_url
from utils import utils
from utils.engine import train_one_epoch, evaluate
from utils.dcn import DeformableConv2d
# float_formatter = "{:.2f}".forma

def _get_timestep_embedding(timesteps, embedding_dim):
        """
        This matches the implementation in Denoising Diffusion Probabilistic Models:
        From Fairseq.
        Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
#         assert len(timesteps.shape) == 1

        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        emb = emb.to(device=timesteps.device)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
        return emb    
    
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * 4, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def se_resnet50(num_classes=1000, pretrained=True):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    backbone = ResNet(SEBottleneck, [3, 4, 6, 3])
    if pretrained:
        backbone.load_state_dict(load_state_dict_from_url(
        "https://github.com/moskomule/senet.pytorch/releases/download/archive/seresnet50-60a8950a85b2b.pkl"))
    # backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # return_layers = {'layer1': 0, 'layer2': 1, 'layer3': 2, 'layer4': 3}
    return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
    in_channels_list = [256, 512, 1024, 2048]
    out_channels = 256
    model = BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model  

def FCOS_model():
    
    backbone = torchvision.models.detection.fcos_resnet50_fpn(weights=FCOS_ResNet50_FPN_Weights.DEFAULT,trainable_backbone_layers=4)
    num_classes = 4
    backbone.out_channels = num_classes
    backbone.backbone = se_resnet50()

        
    return backbone

def postprocess_detections(self, head_outputs, anchors, image_shapes):
        class_logits = head_outputs["cls_logits"]
        box_regression = head_outputs["bbox_regression"]
        box_ctrness = head_outputs["bbox_ctrness"]

        num_images = len(image_shapes)

        detections = []

        for index in range(num_images):
            box_regression_per_image = [br[index] for br in box_regression]
            logits_per_image = [cl[index] for cl in class_logits]
            box_ctrness_per_image = [bc[index] for bc in box_ctrness]
            anchors_per_image, image_shape = anchors[index], image_shapes[index]

            image_boxes = []
            image_scores = []
            image_labels = []

            for box_regression_per_level, logits_per_level, box_ctrness_per_level, anchors_per_level in zip(
                box_regression_per_image, logits_per_image, box_ctrness_per_image, anchors_per_image
            ):
                num_classes = logits_per_level.shape[-1]

                # remove low scoring boxes
                scores_per_level = torch.sqrt(
                    torch.sigmoid(logits_per_level) * torch.sigmoid(box_ctrness_per_level)
                ).flatten()
                keep_idxs = scores_per_level > self.score_thresh
                scores_per_level = scores_per_level[keep_idxs]
                topk_idxs = torch.where(keep_idxs)[0]

                # keep only topk scoring predictions
                num_topk = det_utils._topk_min(topk_idxs, self.topk_candidates, 0)
                scores_per_level, idxs = scores_per_level.topk(num_topk)
                topk_idxs = topk_idxs[idxs]

                anchor_idxs = torch.div(topk_idxs, num_classes, rounding_mode="floor")
                labels_per_level = topk_idxs % num_classes

                boxes_per_level = self.box_coder.decode_single(
                    box_regression_per_level[anchor_idxs], anchors_per_level[anchor_idxs]
                )
                boxes_per_level = box_ops.clip_boxes_to_image(boxes_per_level, image_shape)

                image_boxes.append(boxes_per_level)
                image_scores.append(scores_per_level)
                image_labels.append(labels_per_level)

            image_boxes = torch.cat(image_boxes, dim=0)
            image_scores = torch.cat(image_scores, dim=0)
            image_labels = torch.cat(image_labels, dim=0)

            # non-maximum suppression
            keep = box_ops.batched_nms(image_boxes, image_scores, image_labels, self.nms_thresh)
            keep = keep[: self.detections_per_img]

            detections.append(
                {
                    "boxes": image_boxes[keep],
                    "scores": image_scores[keep],
                    "labels": image_labels[keep],
                }
            )

        return detections
    
def head_compute_loss(
        self,
        targets,
        head_outputs,
        anchors,
        matched_idxs,
    ):

        cls_logits = head_outputs["cls_logits"]  # [N, HWA, C]
        bbox_regression = head_outputs["bbox_regression"]  # [N, HWA, 4]
        bbox_ctrness = head_outputs["bbox_ctrness"]  # [N, HWA, 1]

        all_gt_classes_targets = []
        all_gt_boxes_targets = []
        for targets_per_image, matched_idxs_per_image in zip(targets, matched_idxs):
            if len(targets_per_image["labels"]) == 0:
                gt_classes_targets = targets_per_image["labels"].new_zeros((len(matched_idxs_per_image),))
                gt_boxes_targets = targets_per_image["boxes"].new_zeros((len(matched_idxs_per_image), 4))
            else:
                gt_classes_targets = targets_per_image["labels"][matched_idxs_per_image.clip(min=0)]
                gt_boxes_targets = targets_per_image["boxes"][matched_idxs_per_image.clip(min=0)]
            gt_classes_targets[matched_idxs_per_image < 0] = -1  # backgroud
            all_gt_classes_targets.append(gt_classes_targets)
            all_gt_boxes_targets.append(gt_boxes_targets)

        all_gt_classes_targets = torch.stack(all_gt_classes_targets)
        # compute foregroud
        foregroud_mask = all_gt_classes_targets >= 0
        num_foreground = foregroud_mask.sum().item()

        # classification loss
        gt_classes_targets = torch.zeros_like(cls_logits)
        gt_classes_targets[foregroud_mask, all_gt_classes_targets[foregroud_mask]] = 1.0
        loss_cls = torchvision.ops.sigmoid_focal_loss(cls_logits, gt_classes_targets, reduction="sum")

        # regression loss: GIoU loss
        # TODO: vectorize this instead of using a for loop
        pred_boxes = [
            self.box_coder.decode_single(bbox_regression_per_image, anchors_per_image)
            for anchors_per_image, bbox_regression_per_image in zip(anchors, bbox_regression)
        ]
        # amp issue: pred_boxes need to convert float
        loss_bbox_reg = torchvision.ops.generalized_box_iou_loss(
            torch.stack(pred_boxes)[foregroud_mask].float(),
            torch.stack(all_gt_boxes_targets)[foregroud_mask],
            reduction="sum",
        )
        loss_bbox_reg_dis = torchvision.ops.distance_box_iou_loss(
            torch.stack(pred_boxes)[foregroud_mask].float(),
            torch.stack(all_gt_boxes_targets)[foregroud_mask],
            reduction="sum",
        )
#         print(loss_bbox_reg_dis)
        
        loss_bbox_reg = loss_bbox_reg

        # ctrness loss
        bbox_reg_targets = [
            self.box_coder.encode_single(anchors_per_image, boxes_targets_per_image)
            for anchors_per_image, boxes_targets_per_image in zip(anchors, all_gt_boxes_targets)
        ]
        bbox_reg_targets = torch.stack(bbox_reg_targets, dim=0)
        if len(bbox_reg_targets) == 0:
            gt_ctrness_targets = bbox_reg_targets.new_zeros(bbox_reg_targets.size()[:-1])
        else:
            left_right = bbox_reg_targets[:, :, [0, 2]]
            top_bottom = bbox_reg_targets[:, :, [1, 3]]
            gt_ctrness_targets = torch.sqrt(
                (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0])
                * (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
            )
        pred_centerness = bbox_ctrness.squeeze(dim=2)
        loss_bbox_ctrness = nn.functional.binary_cross_entropy_with_logits(
            pred_centerness[foregroud_mask], gt_ctrness_targets[foregroud_mask], reduction="sum"
        )

        return {
            "classification": loss_cls / max(1, num_foreground),
            "bbox_regression": loss_bbox_reg / max(1, num_foreground),
            "bbox_ctrness": loss_bbox_ctrness / max(1, num_foreground),
        }

def compute_loss(
        self,
        targets,
        head_outputs,
        anchors,
        num_anchors_per_level,
    ):
        matched_idxs = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            if targets_per_image["boxes"].numel() == 0:
                matched_idxs.append(
                    torch.full((anchors_per_image.size(0),), -1, dtype=torch.int64, device=anchors_per_image.device)
                )
                continue

            gt_boxes = targets_per_image["boxes"]
            gt_centers = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2  # Nx2
            anchor_centers = (anchors_per_image[:, :2] + anchors_per_image[:, 2:]) / 2  # N
            anchor_sizes = anchors_per_image[:, 2] - anchors_per_image[:, 0]
            # center sampling: anchor point must be close enough to gt center.
            pairwise_match = (anchor_centers[:, None, :] - gt_centers[None, :, :]).abs_().max(
                dim=2
            ).values < self.center_sampling_radius * anchor_sizes[:, None]
            # compute pairwise distance between N points and M boxes
            x, y = anchor_centers.unsqueeze(dim=2).unbind(dim=1)  # (N, 1)
            x0, y0, x1, y1 = gt_boxes.unsqueeze(dim=0).unbind(dim=2)  # (1, M)
            pairwise_dist = torch.stack([x - x0, y - y0, x1 - x, y1 - y], dim=2)  # (N, M)

            # anchor point must be inside gt
            pairwise_match &= pairwise_dist.min(dim=2).values > 0

            # each anchor is only responsible for certain scale range.
            lower_bound = anchor_sizes * 4
            lower_bound[: num_anchors_per_level[0]] = 0
            upper_bound = anchor_sizes * 8
            upper_bound[-num_anchors_per_level[-1] :] = float("inf")
            pairwise_dist = pairwise_dist.max(dim=2).values
            pairwise_match &= (pairwise_dist > lower_bound[:, None]) & (pairwise_dist < upper_bound[:, None])

            # match the GT box with minimum area, if there are multiple GT matches
            gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])  # N
            pairwise_match = pairwise_match.to(torch.float32) * (1e8 - gt_areas[None, :])
            min_values, matched_idx = pairwise_match.max(dim=1)  # R, per-anchor match
            matched_idx[min_values < 1e-5] = -1  # unmatched anchors are assigned -1

            matched_idxs.append(matched_idx)

        return head_compute_loss(self, targets, head_outputs, anchors, matched_idxs)
    
# AttentionFPN MID1
# AttentionFPN MID1
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # backbone
        model = FCOS_model()
        self.backbone = model.backbone
#         self.fpn = model.fpn
        self.anchor_generator = model.anchor_generator
        self.head = model.head
        self.transform = model.transform
        
#         self.compute_loss = torchvision.models.detection.fcos.FCOS.compute_loss
        self.compute_loss = compute_loss
        self.postprocess_detections = postprocess_detections
        self.eager_outputs = torchvision.models.detection.fcos.FCOS.eager_outputs
        self.min_size = 800
        self.max_size = 1333
        self.center_sampling_radius = 1.5
        self.score_thresh = 0.2
        self.topk_candidates= 800
        self.nms_thresh = 0.6
        self.detections_per_img= 100
        self.box_coder = det_utils.BoxLinearCoder(normalize_by_size=True)
    
        self.ATconv1 = nn.Sequential(
            nn.Conv2d(3, 16, 7, 2, 3),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1,inplace=False),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=False),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1,inplace=False),
        )
        self.ATconv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 2, 0),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1,inplace=False),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1,inplace=False),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1,inplace=False),
        )
        
        
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        
        self.up = nn.Linear(80, 64)
#         self.relu = nn.ReLU()
        self.down = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
        
        # deform after backbone
#         conv = DeformableConv2d
        self.deform_conv = nn.Sequential(
            DeformableConv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1,inplace=False)
        )
        self.deform_conv1 = nn.Sequential(
            DeformableConv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1,inplace=False)
        )
        self.deform_conv2 = nn.Sequential(
            DeformableConv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1,inplace=False)
        )

        
    def forward(self, images, images_2, times, loss_weights = None, targets = None):
        # transform the input
        original_image_sizes = []
        batch_size = len(images)
#         print(batch_size)
        for img in images:
            val = img.shape[-2:]
            original_image_sizes.append((val[0], val[1]))
        
        #ImageList
        images_mix, _ = self.transform(images, targets)
        images, targets = self.transform(images, targets)
        images_2, _ = self.transform(images_2, targets)
        
        images_mix_tensor = images_mix.tensors
    
    #SE-Block 
    #================================================
        images_tensor = images.tensors
#         print('img1 : ', images_tensor.shape)
        images_2_tensor = images_2.tensors
#         print('img2 : ', images_2_tensor.shape)
        
        cat_feature = torch.stack((images_tensor,images_2_tensor),1)
        _, _, _, h_1, w_1 = cat_feature.shape
        cat_feature = cat_feature.reshape(batch_size*2,3,h_1,w_1)
#         print(cat_feature.shape)
        
        # Attention Part 8, 3, 1024, 1024
        attention_f1 = self.ATconv1(cat_feature)
#         print(attention_f1.shape)
        
        attention_f2 = self.ATconv2(attention_f1)
#         print(attention_f1.shape)
        
        sq_f2 = self.squeeze(attention_f2)
#         print(sq_f1.shape)
        # 8, 16, 1, 1

        # 4, 32
        att_featurs = sq_f2.reshape(batch_size,64)
        # 4, 32
        times_emb = _get_timestep_embedding(times,16)
#         print(at_weight.shape)
        
        features_add_temb = torch.cat((att_featurs, times_emb), dim=1) 
#         print(at_weight_2.shape)
        at_weight = self.up(features_add_temb)
#         at_weight = self.up(att_featurs)
        # 4, 1
        at_weight = self.down(at_weight)
        at_weight = at_weight.reshape(batch_size)
#         print(at_weight.shape)
        at_weight_sigm = self.sigmoid(at_weight)
        
#         print('sigm: ', at_weight_sigm)
        
        
        for i in range(batch_size):
            images_mix_tensor[i] = images_tensor[i] + images_2_tensor[i] * (at_weight_sigm[i])
    #================================================
#         print('img mix : ', images_tensor[0][0][0][:50])
        
#         features = self.backbone(images.tensors)
        features = self.backbone(images_mix_tensor)
        features['0'] = self.deform_conv(features['0'])
        features['1'] = self.deform_conv1(features['1'])
        features['2'] = self.deform_conv2(features['2'])
    
        
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        features = list(features.values())
        head_outputs = self.head(features)
        anchors = self.anchor_generator(images, features)
        num_anchors_per_level = [x.size(2) * x.size(3) for x in features]
        
        losses = {}
        detections = [{}]
        
        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training mode")
            else:
                # compute the losses
#                 print(loss_weights)
                loss_weights_avg = loss_weights.mean()
                losses = self.compute_loss(self, targets, head_outputs, anchors, num_anchors_per_level)
#                 print(losses)
                losses['bbox_regression'] = losses['bbox_regression']*loss_weights_avg
                # print(losses)
#                 losses.

        else:
            # split outputs per level
            split_head_outputs= {}
            for k in head_outputs:
                split_head_outputs[k] = list(head_outputs[k].split(num_anchors_per_level, dim=1))
            split_anchors = [list(a.split(num_anchors_per_level)) for a in anchors]

            # compute the detections
            detections = self.postprocess_detections(self,split_head_outputs, split_anchors, images.image_sizes)
            detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)


        return self.eager_outputs(self, losses, detections)