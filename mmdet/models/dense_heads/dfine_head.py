import copy
import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmcv.cnn import Linear
from mmengine.model import constant_init
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.models.losses import QualityFocalLoss
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh
from mmdet.utils import InstanceList, reduce_mean
from ..layers import inverse_sigmoid
from .atss_vlfusion_head import convert_grounding_to_cls_scores
from .dino_head import DINOHead
from .grounding_dino_head import GroundingDINOHead



from .dfine_utils.dfine_utils import weighting_function, distance2bbox
from .dfine_utils.criterion import DFINECriterion

import torch.nn.functional as F
class Integral(nn.Module):
    """
    A static layer that calculates integral results from a distribution.

    This layer computes the target location using the formula: `sum{Pr(n) * W(n)}`,
    where Pr(n) is the softmax probability vector representing the discrete
    distribution, and W(n) is the non-uniform Weighting Function.

    Args:
        reg_max (int): Max number of the discrete bins. Default is 32.
                       It can be adjusted based on the dataset or task requirements.
    """

    def __init__(self, reg_max=32):
        super(Integral, self).__init__()
        self.reg_max = reg_max

    def forward(self, x, project):
        shape = x.shape
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = F.linear(x, project.to(x.device)).reshape(-1, 4)
        return x.reshape(list(shape[:-1]) + [-1])

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act="relu"):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.act = nn.ReLU(act)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class LQE(nn.Module):
    def __init__(self, k, hidden_dim, num_layers, reg_max):
        super(LQE, self).__init__()
        self.k = k
        self.reg_max = reg_max
        self.reg_conf = MLP(4 * (k + 1), hidden_dim, 1, num_layers)
        nn.init.constant_(self.reg_conf.layers[-1].bias, 0)
        nn.init.constant_(self.reg_conf.layers[-1].weight, 0)

    def forward(self, scores, pred_corners):
        B, L, _ = pred_corners.size()
        prob = F.softmax(pred_corners.reshape(B, L, 4, self.reg_max + 1), dim=-1)
        prob_topk, _ = prob.topk(self.k, dim=-1)
        stat = torch.cat([prob_topk, prob_topk.mean(dim=-1, keepdim=True)], dim=-1)
        quality_score = self.reg_conf(stat.reshape(B, L, -1))
        return scores + quality_score
    
# @MODELS.register_module()
class DFINEHead(GroundingDINOHead):
    def __init__(self,eval_idx=-1,layer_scale=1, **kwargs):
        super().__init__(**kwargs)
        self.reg_max = 32
        self.up = nn.Parameter(torch.tensor([0.5]), requires_grad=False)
        self.reg_scale = nn.Parameter(torch.tensor([4.0]), requires_grad=False)
        self.project = weighting_function(self.reg_max, self.up, self.reg_scale)
        self.integral = Integral(self.reg_max)
        self.eval_idx = eval_idx if eval_idx >= 0 else self.num_pred_layer + eval_idx
        self.scaled_dims = round(layer_scale * self.embed_dims)
        self.criterion = DFINECriterion(num_classes=11)
    
    def _init_layers(self) -> None:
        """Initialize classification branch and regression branch of head."""
        fc_cls = MLP(self.embed_dims, self.embed_dims, self.num_classes, 3)
        reg_branch = []
        
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, 4 * (self.reg_max + 1)))
        reg_branch = nn.Sequential(*reg_branch)

        
        # NOTE: due to the fc_cls is a contrastive embedding and don't
        # have any trainable parameters,we do not need to copy it.
        #classification branch
        self.cls_branches = nn.ModuleList(
            [nn.Linear(self.embed_dims, self.num_classes) for _ in range(self.eval_idx + 1)]
            + [nn.Linear(self.scaled_dims, self.num_classes) for _ in range(self.num_pred_layer - self.eval_idx - 1)]
        )
        #regression branch
        self.reg_branches = nn.ModuleList(
            [
                MLP(self.embed_dims, self.embed_dims, 4 * (self.reg_max + 1), 3)
                for _ in range(self.eval_idx + 1)
            ] + [
                MLP(self.scaled_dims, self.scaled_dims, 4 * (self.reg_max + 1), 3)
                for _ in range(self.num_pred_layer - self.eval_idx - 1)
            ]
        )
        # pre regression branch
        self.pre_reg_branches = MLP(self.embed_dims, self.embed_dims, 4, 3)
        #location quality estimation
        self.lqe_branches = nn.ModuleList(
            [LQE(4, 64, 2, self.reg_max) for _ in range(self.num_pred_layer)]
        )
            
    def forward(
        self,
        hidden_states: Tensor,
        references: List[Tensor],
        
    ) -> Tuple[Tensor]:
        
        all_layers_outputs_classes = []
        all_layers_outputs_coords = []
        all_layers_outputs_corners = []
        all_layers_outputs_refs = []
        pred_corners_undetach = 0
        
        for layer_id in range(hidden_states.shape[0]):
            reference = inverse_sigmoid(references[layer_id])
            # NOTE The last reference will not be used.
            hidden_state = hidden_states[layer_id]
            
            if layer_id == 0:
                pre_bboxes = F.sigmoid(self.pre_bbox_head(hidden_state) + reference)
                pre_scores = self.cls_branches[0](hidden_state)
                ref_points_initial = pre_bboxes.detach()

            # Refine bounding box corners using FDR, integrating previous layer's corrections
            pred_corners = self.reg_branches[layer_id](hidden_state) + pred_corners_undetach
            outputs_coord = distance2bbox(
                ref_points_initial, self.integral(pred_corners, self.project), self.reg_scale
            )
            
            # class branch
            outputs_class = self.cls_branches[layer_id](hidden_state)
            # Lqe does not affect the performance here.
            outputs_class = self.lqe_layers[layer_id](outputs_class, pred_corners)
        
            all_layers_outputs_classes.append(outputs_class)
            all_layers_outputs_coords.append(outputs_coord)
            all_layers_outputs_corners.append(pred_corners)
            all_layers_outputs_refs.append(reference)
            # iterate
            pred_corners_undetach = pred_corners

        all_layers_outputs_classes = torch.stack(all_layers_outputs_classes)
        all_layers_outputs_coords = torch.stack(all_layers_outputs_coords)
        all_layers_outputs_corners = torch.stack(pred_corners_undetach)
        all_layers_outputs_refs = torch.stack(all_layers_outputs_refs)
        
        return (all_layers_outputs_classes,
                all_layers_outputs_coords,
                all_layers_outputs_corners,
                all_layers_outputs_refs,
                pre_bboxes,
                pre_scores)
        
    def predict(self,
            hidden_states: Tensor,
            references: List[Tensor],
            batch_data_samples: SampleList,
            rescale: bool = True) -> InstanceList:
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        batch_token_positive_maps = [
            data_samples.token_positive_map
            for data_samples in batch_data_samples
        ]

        all_layers_outputs_classes,all_layers_outputs_coords,all_layers_outputs_corners,all_layers_outputs_refs,pre_bboxes,pre_scores = self(hidden_states, references)

        predictions = self.predict_by_feat(
            all_layers_outputs_classes,
            all_layers_outputs_coords,
            batch_img_metas=batch_img_metas,
            batch_token_positive_maps=batch_token_positive_maps,
            rescale=rescale)
        return predictions
    
    def loss(self, hidden_states: Tensor, references: List[Tensor],
            enc_outputs_class: Tensor, enc_outputs_coord: Tensor,
            batch_data_samples: SampleList, dn_meta: Dict[str, int]) -> dict:

        batch_gt_instances = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)

        outputs = self(hidden_states, references)
        # outputs:all_layers_outputs_classes,all_layers_outputs_coords,all_layers_outputs_corners,all_layers_outputs_refs,pre_bboxes,pre_scores
        loss_inputs = outputs + (enc_outputs_class, enc_outputs_coord,
                              batch_gt_instances, batch_img_metas, dn_meta)
        losses = self.loss_by_feat(*loss_inputs)
        
        
        targets = []
        for data_samples in batch_data_samples:
            targets.append(dict(boxes=data_samples.gt_instances.bboxes,
                          labels=data_samples.gt_instances.labels,
                          image_id = data_samples.img_id,
                          orig_size= data_samples.orisize))
        
        
        
        add_losses = {}
        
        losses.update(add_losses)
        
        
        return losses

    # def loss_by_feat(
    #     self,
    #     all_layers_cls_scores: Tensor,
    #     all_layers_bbox_preds: Tensor,
    #     enc_cls_scores: Tensor,
    #     enc_bbox_preds: Tensor,
    #     batch_gt_instances: InstanceList,
    #     batch_img_metas: List[dict],
    #     dn_meta: Dict[str, int],
    #     batch_gt_instances_ignore= None
    # ) -> Dict[str, Tensor]:
    #     """Loss function.

    #     Args:
    #         all_layers_cls_scores (Tensor): Classification scores of all
    #             decoder layers, has shape (num_decoder_layers, bs,
    #             num_queries_total, cls_out_channels), where
    #             `num_queries_total` is the sum of `num_denoising_queries`
    #             and `num_matching_queries`.
    #         all_layers_bbox_preds (Tensor): Regression outputs of all decoder
    #             layers. Each is a 4D-tensor with normalized coordinate format
    #             (cx, cy, w, h) and has shape (num_decoder_layers, bs,
    #             num_queries_total, 4).
    #         enc_cls_scores (Tensor): The score of each point on encode
    #             feature map, has shape (bs, num_feat_points, cls_out_channels).
    #         enc_bbox_preds (Tensor): The proposal generate from the encode
    #             feature map, has shape (bs, num_feat_points, 4) with the last
    #             dimension arranged as (cx, cy, w, h).
    #         batch_gt_instances (list[:obj:`InstanceData`]): Batch of
    #             gt_instance. It usually includes ``bboxes`` and ``labels``
    #             attributes.
    #         batch_img_metas (list[dict]): Meta information of each image, e.g.,
    #             image size, scaling factor, etc.
    #         dn_meta (Dict[str, int]): The dictionary saves information about
    #             group collation, including 'num_denoising_queries' and
    #             'num_denoising_groups'. It will be used for split outputs of
    #             denoising and matching parts and loss calculation.
    #         batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
    #             Batch of gt_instances_ignore. It includes ``bboxes`` attribute
    #             data that is ignored during training and testing.
    #             Defaults to None.

    #     Returns:
    #         dict[str, Tensor]: A dictionary of loss components.
    #     """
    #     # extract denoising and matching part of outputs
    #     (all_layers_matching_cls_scores, all_layers_matching_bbox_preds,
    #      all_layers_denoising_cls_scores, all_layers_denoising_bbox_preds) = \
    #         self.split_outputs(
    #             all_layers_cls_scores, all_layers_bbox_preds, dn_meta)

    #     loss_dict = super(DeformableDETRHead, self).loss_by_feat(
    #         all_layers_matching_cls_scores, all_layers_matching_bbox_preds,
    #         batch_gt_instances, batch_img_metas, batch_gt_instances_ignore)
    #     # NOTE DETRHead.loss_by_feat but not DeformableDETRHead.loss_by_feat
    #     # is called, because the encoder loss calculations are different
    #     # between DINO and DeformableDETR.

    #     # loss of proposal generated from encode feature map.
    #     if enc_cls_scores is not None:
    #         # NOTE The enc_loss calculation of the DINO is
    #         # different from that of Deformable DETR.
    #         enc_loss_cls, enc_losses_bbox, enc_losses_iou = \
    #             self.loss_by_feat_single(
    #                 enc_cls_scores, enc_bbox_preds,
    #                 batch_gt_instances=batch_gt_instances,
    #                 batch_img_metas=batch_img_metas)
    #         loss_dict['enc_loss_cls'] = enc_loss_cls
    #         loss_dict['enc_loss_bbox'] = enc_losses_bbox
    #         loss_dict['enc_loss_iou'] = enc_losses_iou

    #     if all_layers_denoising_cls_scores is not None:
    #         # calculate denoising loss from all decoder layers
    #         dn_losses_cls, dn_losses_bbox, dn_losses_iou = self.loss_dn(
    #             all_layers_denoising_cls_scores,
    #             all_layers_denoising_bbox_preds,
    #             batch_gt_instances=batch_gt_instances,
    #             batch_img_metas=batch_img_metas,
    #             dn_meta=dn_meta)
    #         # collate denoising loss
    #         loss_dict['dn_loss_cls'] = dn_losses_cls[-1]
    #         loss_dict['dn_loss_bbox'] = dn_losses_bbox[-1]
    #         loss_dict['dn_loss_iou'] = dn_losses_iou[-1]
    #         for num_dec_layer, (loss_cls_i, loss_bbox_i, loss_iou_i) in \
    #                 enumerate(zip(dn_losses_cls[:-1], dn_losses_bbox[:-1],
    #                               dn_losses_iou[:-1])):
    #             loss_dict[f'd{num_dec_layer}.dn_loss_cls'] = loss_cls_i
    #             loss_dict[f'd{num_dec_layer}.dn_loss_bbox'] = loss_bbox_i
    #             loss_dict[f'd{num_dec_layer}.dn_loss_iou'] = loss_iou_i
    #     return loss_dict