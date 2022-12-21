import math
import torch
import torchvision

from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
from torchvision.ops.boxes import batched_nms

import torch
from torch import nn

# point generator
from .point_generator import PointGenerator

# input / output transforms
from .transforms import GeneralizedRCNNTransform

# loss functions
from .losses import sigmoid_focal_loss, giou_loss


class FCOSClassificationHead(nn.Module):
    """
    A classification head for FCOS with convolutions and group norms

    Args:
        in_channels (int): number of channels of the input feature.
        num_classes (int): number of classes to be predicted
        num_convs (Optional[int]): number of conv layer. Default: 2.
        prior_probability (Optional[float]): probability of prior. Default: 0.01.
    """

    def __init__(self, in_channels, num_classes, num_convs=2, prior_probability=0.01):
        super().__init__()
        self.num_classes = num_classes

        conv = []
        for _ in range(num_convs):
            conv.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            conv.append(nn.GroupNorm(16, in_channels))
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

        # A separate background category is not needed, as later we will consider
        # C binary classfication problems here (using sigmoid focal loss)
        self.cls_logits = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1, padding=1
        )
        torch.nn.init.normal_(self.cls_logits.weight, std=0.01)
        # see Sec 3.3 in "Focal Loss for Dense Object Detection'
        torch.nn.init.constant_(
            self.cls_logits.bias, -math.log((1 - prior_probability) / prior_probability)
        )

    def forward(self, x):
        """
        Fill in the missing code here. The head will be applied to all levels
        of the feature pyramid, and predict a single logit for each location on
        every feature location.

        Without pertumation, the results will be a list of tensors in increasing
        depth order, i.e., output[0] will be the feature map with highest resolution
        and output[-1] will the featuer map with lowest resolution. The list length is
        equal to the number of pyramid levels. Each tensor in the list will be
        of size N x C x H x W, storing the classification logits (scores).

        Some re-arrangement of the outputs is often preferred for training / inference.
        You can choose to do it here, or in compute_loss / inference.
        """
        output = []
        for features in x:
          logits = self.conv(features)
          output.append(self.cls_logits(logits))
        return output


class FCOSRegressionHead(nn.Module):
    """
    A regression head for FCOS with convolutions and group norms.
    This head predicts
    (a) the distances from each location (assuming foreground) to a box
    (b) a center-ness score

    Args:
        in_channels (int): number of channels of the input feature.
        num_convs (Optional[int]): number of conv layer. Default: 2.
    """

    def __init__(self, in_channels, num_convs=2):
        super().__init__()
        conv = []
        for _ in range(num_convs):
            conv.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            conv.append(nn.GroupNorm(16, in_channels))
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

        # regression outputs must be positive
        self.bbox_reg = nn.Sequential(
            nn.Conv2d(in_channels, 4, kernel_size=3, stride=1, padding=1), nn.ReLU()
        )
        self.bbox_ctrness = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1, padding=1
        )

        self.apply(self.init_weights)
        # The following line makes sure the regression head output a non-zero value.
        # If your regression loss remains the same, try to uncomment this line.
        # It helps the initial stage of training
        # torch.nn.init.normal_(self.bbox_reg[0].bias, mean=1.0, std=0.1)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.normal_(m.weight, std=0.01)
            torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Fill in the missing code here. The logic is rather similar to
        FCOSClassificationHead. The key difference is that this head bundles both
        regression outputs and the center-ness scores.

        Without pertumation, the results will be two lists of tensors in increasing
        depth order, corresponding to regression outputs and center-ness scores.
        Again, the list length is equal to the number of pyramid levels.
        Each tensor in the list will of size N x 4 x H x W (regression)
        or N x 1 x H x W (center-ness).

        Some re-arrangement of the outputs is often preferred for training / inference.
        You can choose to do it here, or in compute_loss / inference.
        """
        out_regress = []
        out_centerness = []

        for feature in x:
          logits = self.conv(feature)
          out_regress.append(self.bbox_reg(logits))
          out_centerness.append(self.bbox_ctrness(logits))

        return out_regress, out_centerness


class FCOS(nn.Module):
    """
    Implementation of (simplified) Fully Convolutional One-Stage object detector,
    as desribed in the journal paper: https://arxiv.org/abs/2006.09214

    Args:
        backbone (string): backbone network, only ResNet18 is supported
        backbone_out_feats (List[string]): output feature maps from the backbone network
        backbone_out_feats_dims (List[int]): backbone output features dimensions
        (in increasing depth order)

        fpn_feats_dim (int): output feature dimension from FPN in increasing depth order
        fpn_strides (List[int]): feature stride for each pyramid level in FPN
        num_classes (int): number of output classes of the model (excluding the background)
        regression_range (List[Tuple[int, int]]): box regression range on each level of the pyramid
        in increasing depth order. E.g., [[0, 32], [32 64]] means that the first level
        of FPN (highest feature resolution) will predict boxes with width and height in range of [0, 32],
        and the second level in the range of [32, 64].

        img_min_size (List[int]): minimum sizes of the image to be rescaled before feeding it to the backbone
        img_max_size (int): maximum size of the image to be rescaled before feeding it to the backbone
        img_mean (Tuple[float, float, float]): mean values used for input normalization.
        img_std (Tuple[float, float, float]): std values used for input normalization.

        train_cfg (Dict): dictionary that specifies training configs, including
            center_sampling_radius (int): radius of the "center" of a groundtruth box,
            within which all anchor points are labeled positive.

        test_cfg (Dict): dictionary that specifies test configs, including
            score_thresh (float): Score threshold used for postprocessing the detections.
            nms_thresh (float): NMS threshold used for postprocessing the detections.
            detections_per_img (int): Number of best detections to keep after NMS.
            topk_candidates (int): Number of best detections to keep before NMS.

        * If a new parameter is added in config.py or yaml file, they will need to defined here.
    """

    def __init__(
        self,
        backbone,
        backbone_out_feats,
        backbone_out_feats_dims,
        fpn_feats_dim,
        fpn_strides,
        num_classes,
        regression_range,
        img_min_size,
        img_max_size,
        img_mean,
        img_std,
        train_cfg,
        test_cfg,
    ):
        super().__init__()
        assert backbone == "ResNet18"
        self.backbone_name = backbone
        self.fpn_strides = fpn_strides
        self.num_classes = num_classes
        self.regression_range = regression_range

        return_nodes = {}
        for feat in backbone_out_feats:
            return_nodes.update({feat: feat})

        # backbone network (resnet18)
        self.backbone = create_feature_extractor(
            resnet18(weights=ResNet18_Weights.DEFAULT), return_nodes=return_nodes
        )

        # feature pyramid network (FPN)
        self.fpn = FeaturePyramidNetwork(
            backbone_out_feats_dims,
            out_channels=fpn_feats_dim,
        )

        # point generator will create a set of points on the 2D image plane
        self.point_generator = PointGenerator(
            img_max_size, fpn_strides, regression_range
        )

        # classification and regression head
        self.cls_head = FCOSClassificationHead(fpn_feats_dim, num_classes)
        self.reg_head = FCOSRegressionHead(fpn_feats_dim)

        # image batching, normalization, resizing, and postprocessing
        self.transform = GeneralizedRCNNTransform(
            img_min_size, img_max_size, img_mean, img_std
        )

        # other params for training / inference
        self.center_sampling_radius = train_cfg["center_sampling_radius"]
        self.score_thresh = test_cfg["score_thresh"]
        self.nms_thresh = test_cfg["nms_thresh"]
        self.detections_per_img = test_cfg["detections_per_img"]
        self.topk_candidates = test_cfg["topk_candidates"]

    """
    We will overwrite the train function. This allows us to always freeze
    all batchnorm layers in the backbone, as we won't have sufficient samples in
    each mini-batch to aggregate the bachnorm stats.
    """
    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                if hasattr(module, "weight"):
                    module.weight.requires_grad_(False)
                if hasattr(module, "bias"):
                    module.bias.requires_grad_(False)
            else:
                module.train(mode)
        return self

    """
    The behavior of the forward function changes depending if the model is
    in training or evaluation mode.

    During training, the model expects both the input tensors
    (list of tensors within the range of [0, 1]), as well as a targets
    (list of dictionary), containing:
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in
          ``[x1, y1, x2, y2]`` format, with ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the class label for each ground-truth box
    The model returns a Dict[Tensor] during training, containing the classification, regression
    and centerness losses, as well as a final loss as a summation of all three terms.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows:
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format,
          with ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the predicted labels for each image
        - scores (Tensor[N]): the scores for each prediction

    See also the comments for compute_loss / inference.
    """

    def forward(self, images, targets):
        # sanity check
        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training")
            else:
                for target in targets:
                    boxes = target["boxes"]
                    torch._assert(
                        isinstance(boxes, torch.Tensor),
                        "Expected target boxes to be of type Tensor.",
                    )
                    torch._assert(
                        len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                        f"Expected target boxes of shape [N, 4], got {boxes.shape}.",
                    )

        # record the original image size, this is needed to decode the box outputs
        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            original_image_sizes.append((val[0], val[1]))

        # transform the input
        images, targets = self.transform(images, targets)

        # get the features from the backbone
        # the result will be a dictionary {feature name : tensor}
        features = self.backbone(images.tensors)

        # send the features from the backbone into the FPN
        # the result is converted into a list of tensors (list length = #FPN levels)
        # this list stores features in increasing depth order, each of size N x C x H x W
        # (N: batch size, C: feature channel, H, W: height and width)
        fpn_features = self.fpn(features)
        fpn_features = list(fpn_features.values())

        # classification / regression heads
        cls_logits = self.cls_head(fpn_features)
        reg_outputs, ctr_logits = self.reg_head(fpn_features)

        # 2D points (corresponding to feature locations) of shape H x W x 2
        points, strides, reg_range = self.point_generator(fpn_features)

        # training / inference
        if self.training:
            # training: generate GT labels, and compute the loss
            losses = self.compute_loss(
                targets, points, strides, reg_range, cls_logits, reg_outputs, ctr_logits
            )
            # return loss during training
            return losses

        else:
            # inference: decode / postprocess the boxes
            detections = self.inference(
                points, strides, cls_logits, reg_outputs, ctr_logits, images.image_sizes, fpn_features
            )
            # rescale the boxes to the input image resolution
            detections = self.transform.postprocess(
                detections, images.image_sizes, original_image_sizes
            )
            # return detectrion results during inference
            return detections

    """
    Fill in the missing code here. This is probably the most tricky part
    in this assignment. Here you will need to compute the object label for each point
    within the feature pyramid. If a point lies around the center of a foreground object
    (as controlled by self.center_sampling_radius), its regression and center-ness
    targets will also need to be computed.

    Further, three loss terms will be attached to compare the model outputs to the
    desired targets (that you have computed), including
    (1) classification (using sigmoid focal for all points)
    (2) regression loss (using GIoU and only on foreground points)
    (3) center-ness loss (using binary cross entropy and only on foreground points)

    Some of the implementation details that might not be obvious
    * The output regression targets are divided by the feature stride (Eq 1 in the paper)
    * All losses are normalized by the number of positive points (Eq 2 in the paper)

    The output must be a dictionary including the loss values
    {
        "cls_loss": Tensor (1)
        "reg_loss": Tensor (1)
        "ctr_loss": Tensor (1)
        "final_loss": Tensor (1)
    }
    where the final_loss is a sum of the three losses and will be used for training.
    """

    def compute_loss(
        self, targets, points, strides, reg_range, cls_logits, reg_outputs, ctr_logits
    ):
        # points # HxWx2
        # list (batchsize), Number of points for all feature-levels --> GT-box Id
        matched_ids = []

        torch._assert(isinstance(target,list), "Model.py compute_loss: Expected target boxes to be of type List.")
        

        all_gt_boxes_targets = []
        all_gt_classes_targets = []
        
        all_gt_reg_out = []
        all_gt_classes_targets = []

        for tid,target in enumerate(targets):
          gt_boxes = target['boxes']  # Mx4
          gt_centers = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2  # Mx2 (M boxes)
          
          per_stride_gt_classes_targets = []
          per_stride_gt_boxes_targets = []
          per_stride_gt_reg_out = []
          per_stride_gt_ctrness_targets = []

          for i,stride in enumerate(strides):
            
            pred = points[i].view(-1,2)   # HWx2,  or Nx2 (N anchors)
            pairwise_match = pred[:,None,:] - gt_centers[None,:,:] # NxMx2

            pairwise_match = pairwise_match.abs_().max(dim=2).values < (self.center_sampling_radius*stride) # NxM

            x, y = pred.unsqueeze(dim=2).unbind(dim=1)  # Nx1
            x0, y0, x1, y1 = gt_boxes.unsqueeze(dim=0).unbind(dim=2)  # 1xM

            paired_dist = torch.stack([x - x0, y - y0, x1 - x, y1 - y], dim=2)
            pairwise_match &= paired_dist.min(dim=2).values > 0   # NxM
            
            lt = (pred[:,None,:] - gt_boxes[:,:2,None])    # N,1,2 - M,2,1 --> N,M,2 
            rb = -(pred[:,None,:] - gt_boxes[:,2:,None])   # N,M,2

            t_dist = torch.cat([lt,rb],dim=2).abs_().max(dim=2).values # N,M
            lower, upper = reg_range[i][0], reg_range[i][1]

            pairwise_match &= (t_dist > lower) & (paired_dist < upper)  # N,M

            # match the GT box with minimum area, if there are multiple GT matches
            gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])  # M
            pairwise_match = pairwise_match.to(torch.float32) * (1e8 - gt_areas[None, :])
            min_values, matched_idx = pairwise_match.max(dim=1)  # R, per-anchor match
            matched_idx[min_values < 1e-5] = -1  # unmatched anchors are assigned -1, (N,)


            gt_classes_targets = target["labels"][matched_idx.clip(min=0)]   # (N,)
            gt_boxes_targets = target["boxes"][matched_idx.clip(min=0)]      # (N,4)
            gt_classes_targets[matched_idx < 0] = -1

            lt_pred = pred - gt_boxes_targets[:,:2]         #(N,2)
            rb_pred = gt_boxes_targets[:,2:] - pred         #(N,2)
            t_pred = torch.cat([lt_pred,rb_pred],dim=-1)/stride     #(N,4)

            left_right = t_pred[:, [0, 2]]
            top_bottom = t_pred[:, [1, 3]]

            gt_ctrness_targets = torch.sqrt(
                (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0])
                * (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
            )          # (N,)


            reg_out = (reg_outputs[i][tid].view(4,-1)).permute(1,0)*stride     # (N(HW),4)
            gt_reg_out = torch.cat([pred-reg_out[:,:2],pred+reg_out[:,2:]],dim=-1)   # HWx4,  or Nx4 (N anchors)
            

            per_stride_gt_classes_targets.append(gt_classes_targets)
            per_stride_gt_boxes_targets.append(gt_boxes_targets)
            per_stride_gt_reg_out.append(gt_reg_out)
            per_stride_gt_ctrness_targets.append(gt_ctrness_targets)

          
          all_gt_classes_targets.append(torch.cat(per_stride_gt_classes_targets,dim=0))  # List, batchsz. (A,1) 
          all_gt_boxes_targets.append(torch.cat(per_stride_gt_boxes_targets,dim=0))          # List, (A,4) 
          all_gt_reg_out(torch.cat(per_stride_gt_reg_out,dim=0))     # list ((A,4) : A = (HW)_1+(HW)_2+(HW)_3
          all_gt_classes_targets.append(torch.cat(per_stride_gt_ctrness_targets),dim=0)


        cls_logits = [t.view(t.shape[0],t.shape[1],-1) for t in cls_logits]
        #reg_outputs = [t.view(t.shape[0],t.shape[1],-1) for t in reg_outputs]
        ctr_logits = [t.view(t.shape[0],t.shape[1],-1) for t in ctr_logits]

        #cls_logits,reg_outputs,ctr_logits = (
        #              torch.cat(cls_logits,dim=2).permute(0,2,1).contiguous(), # (bs,A,C)
        #              torch.cat(reg_outputs,dim=2).permute(0,2,1).contiguous(),# (bs,A,4)
        #              torch.cat(ctr_logits,dim=2).permute(0,2,1).contiguous()) # (bs,A,1)
        
        cls_logits,ctr_logits = (
                      torch.cat(cls_logits,dim=2).permute(0,2,1).contiguous(), # (bs,A,C)
                      torch.cat(ctr_logits,dim=2).permute(0,2,1).contiguous()) # (bs,A,1)


        all_gt_boxes_targets, all_gt_classes_targets = (
            torch.stack(all_gt_boxes_targets),
            torch.stack(all_gt_classes_targets),
            torch.stack(all_gt_reg_out),
            torch.stack(all_gt_classes_targets)
        )      # [bs,A,4], [bs,A,1] , [bs,A,4], [bs,A,1]

        # compute foregroud
        foregroud_mask = all_gt_classes_targets >= 0
        num_foreground = foregroud_mask.sum().item()

        # classification loss
        gt_classes_targets = torch.zeros_like(cls_logits)
        gt_classes_targets[foregroud_mask, all_gt_classes_targets[foregroud_mask]] = 1.0
        cls_loss = sigmoid_focal_loss(cls_logits, gt_classes_targets, reduction="sum")

        # regression loss
        reg_loss = giou_loss(all_gt_reg_out[foregroud_mask],all_gt_boxes_targets[foregroud_mask],reduction='sum')
        
        # centerness loss
        ctr_logits = ctr_logits.squeeze(dim=-1)
        ctr_loss = nn.functional.binary_cross_entropy_with_logits(
            ctr_logits[foregroud_mask], all_gt_classes_targets[foregroud_mask], reduction="sum"
        )

        losses = {}
        losses['cls_loss'] = cls_loss / max(1,num_foreground)
        losses['reg_loss'] = reg_loss / max(1,num_foreground)
        losses['ctr_loss'] = ctr_loss / max(1,num_foreground)
        final_loss = cls_loss + reg_loss + ctr_loss
        losses['final_loss'] = final_loss
        return losses

    """
    Fill in the missing code here. The inference is also a bit involved. It is
    much easier to think about the inference on a single image
    (a) Loop over every pyramid level
        (1) compute the object scores
        (2) deocde the boxes
        (3) only keep boxes with scores larger than self.score_thresh
    (b) Combine all object candidates across levels and keep the top K (self.topk_candidates)
    (c) Remove boxes outside of the image boundaries (due to padding)
    (d) Run non-maximum suppression to remove any duplicated boxes
    (e) keep the top K boxes after NMS (self.detections_per_img)

    Some of the implementation details that might not be obvious
    * As the output regression target is divided by the feature stride during training,
    you will have to multiply the regression outputs by the stride at inference time.
    * Most of the detectors will allow two overlapping boxes from two different categories
    (e.g., one from "shirt", the other from "person"). That means that
        (a) one can decode two same boxes of different categories from one location;
        (b) NMS is only performed within each category.
    * Regression range is not used, as the range is not enforced during inference.
    * image_shapes is needed to remove boxes outside of the images.
    * Output labels needed to be offseted by +1 to compensate for the input label transform

    The output must be a list of dictionary items (one for each image) following
    [
        {
            "boxes": Tensor (N x 4)
            "scores": Tensor (N, )
            "labels": Tensor (N, )
        },
    ]
    """

    def inference(
        self, points, strides, cls_logits, reg_outputs, ctr_logits, image_shapes, fpn_features
    ):  
        detections = []

        # permute bbox regression output from (N, 4, H, W) to (N, HW, 4).
        N, _, H, W = reg_outputs.shape
        reg_outputs = reg_outputs.view(N, -1, 4, H, W)
        reg_outputs = reg_outputs.permute(0, 3, 4, 1, 2)
        reg_outputs = reg_outputs.reshape(N, -1, 4)

        # Permute classification output from (N, C, H, W) to (N, HW, C).
        N, _, H, W = cls_logits.shape
        cls_logits = cls_logits.view(N, -1, self.num_classes, H, W)
        cls_logits = cls_logits.permute(0, 3, 4, 1, 2)
        cls_logits = cls_logits.reshape(N, -1, self.num_classes)

        # permute bbox ctrness output from (N, 1, H, W) to (N, HW, 1).
        ctr_logits = ctr_logits.view(N, -1, 1, H, W)
        ctr_logits = ctr_logits.permute(0, 3, 4, 1, 2)
        ctr_logits = ctr_logits.reshape(N, -1, 1)

        # level sizes
        level_sizes = [feature.size(2) * feature.size(3) for feature in fpn_features]

        # splits per level
        split_reg_outputs = list(reg_outputs.split(level_sizes, dim = 1))
        split_cls_logits = list(cls_logits.split(level_sizes, dim = 1))
        split_ctr_logits = list(ctr_logits.split(level_sizes, dim = 1))

        # looping over every image
        for idx in range(len(image_shapes)):
            # find the regression, classification and correctness score per image
            cls_logits_image = [classification[idx] for classification in split_cls_logits]
            reg_outputs_image = [regression[idx] for regression in split_reg_outputs]
            ctr_logits_image = [centerness[idx] for centerness in split_ctr_logits]
            
            image_boxes = []
            image_scores = []
            image_labels = []

            # loop over all pyramid levels
            for level, reg_outputs_level, cls_logits_level, ctr_logits_level in enumerate(zip(
                reg_outputs_image, cls_logits_image, ctr_logits_image,
            )):
                # compute scores
                scores_level = torch.sqrt(torch.sigmoid(cls_logits_level) * torch.sigmoid(ctr_logits_level)).flatten()
                
                # threshold scores
                scores_level_thresholded = scores_level[scores_level > self.score_thresh]

                # keep only top K candidates
                scores_level_thresholded_top_k, top_k_candidate_indices = scores_level_thresholded.topk(k = self.topk_candidates, dim = 0)

                # get boxes --> TODO
                left = reg_outputs_level[0] * strides[level]
                top = reg_outputs_level[1] * strides[level]
                right = reg_outputs_level[2] * strides[level]
                bottom = reg_outputs_level[3] * strides[level]

                x_0 = points.permute(2,0,1)[0] - left
                x_1 = right + points.permute(2,0,1)[0]
                y_0 = points.permute(2,0,1)[1] - top
                y_1 = bottom + points.permute(2,0,1)[1]

                # clip boxes to stay within image --> TODO
                x_0 = x_0.clamp(min = 0, max = image_shapes[idx][1])
                x_1 = x_1.clamp(min = 0, max = image_shapes[idx][1])
                y_0 = y_0.clamp(min = 0, max = image_shapes[idx][0])
                y_1 = y_1.clamp(min = 0, max = image_shapes[idx][0])
                boxes_level_clipped = torch.stack((boxes_x, boxes_y), dim = boxes_level.dim)
                boxes_level_clipped = boxes_level_clipped.reshape(boxes_level.shape)

                image_boxes.append(boxes_level_clipped)
                image_scores.append(scores_level_thresholded_top_k)
                image_labels.append((top_k_candidate_indices % cls_logits_image[level].shape[0]) + 1)
            
            image_boxes = torch.cat(image_boxes, dim = 0)
            image_scores = torch.cat(image_scores, dim = 0)
            image_labels = torch.cat(image_labels, dim = 0)

            # non-maximum suppression
            keep = batched_nms(image_boxes, image_scores, image_labels, self.nms_thresh)[ : self.detections_per_img]

            detections.append(
                {
                    "boxes": image_boxes[keep],
                    "scores": image_scores[keep],
                    "labels": image_labels[keep],
                }
            )

        return detections
