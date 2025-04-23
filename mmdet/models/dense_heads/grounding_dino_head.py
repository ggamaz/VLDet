# Copyright (c) OpenMMLab. All rights reserved.
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
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh, bbox_overlaps
from mmdet.utils import InstanceList, reduce_mean
from ..layers import inverse_sigmoid
from .atss_vlfusion_head import convert_grounding_to_cls_scores
from .dino_head import DINOHead
from .deformable_detr_head import DeformableDETRHead

class ContrastiveEmbed(nn.Module):
    """text visual ContrastiveEmbed layer.

    Args:
        max_text_len (int, optional): Maximum length of text.
        log_scale (Optional[Union[str, float]]):  The initial value of a
          learnable parameter to multiply with the similarity
          matrix to normalize the output.  Defaults to 0.0.
          - If set to 'auto', the similarity matrix will be normalized by
            a fixed value ``sqrt(d_c)`` where ``d_c`` is the channel number.
          - If set to 'none' or ``None``, there is no normalization applied.
          - If set to a float number, the similarity matrix will be multiplied
            by ``exp(log_scale)``, where ``log_scale`` is learnable.
        bias (bool, optional): Whether to add bias to the output.
          If set to ``True``, a learnable bias that is initialized as -4.6
          will be added to the output. Useful when training from scratch.
          Defaults to False.
    """

    def __init__(self,
                 max_text_len: int = 256,
                 log_scale: Optional[Union[str, float]] = None,
                 bias: bool = False):
        super().__init__()
        self.max_text_len = max_text_len
        self.log_scale = log_scale
        if isinstance(log_scale, float):
            self.log_scale = nn.Parameter(
                torch.Tensor([float(log_scale)]), requires_grad=True)
        elif log_scale not in ['auto', 'none', None]:
            raise ValueError(f'log_scale should be one of '
                             f'"auto", "none", None, but got {log_scale}')

        self.bias = None
        if bias:
            bias_value = -math.log((1 - 0.01) / 0.01)
            self.bias = nn.Parameter(
                torch.Tensor([bias_value]), requires_grad=True)

    def forward(self, visual_feat: Tensor, text_feat: Tensor,
                text_token_mask: Tensor) -> Tensor:
        """Forward function.

        Args:
            visual_feat (Tensor): Visual features.
            text_feat (Tensor): Text features.
            text_token_mask (Tensor): A mask used for text feats.

        Returns:
            Tensor: Classification score.
        """
        res = visual_feat @ text_feat.transpose(-1, -2)
        if isinstance(self.log_scale, nn.Parameter):
            res = res * self.log_scale.exp()
        elif self.log_scale == 'auto':
            # NOTE: similar to the normalizer in self-attention
            res = res / math.sqrt(visual_feat.shape[-1])
        if self.bias is not None:
            res = res + self.bias
        res.masked_fill_(~text_token_mask[:, None, :], float('-inf'))

        new_res = torch.full((*res.shape[:-1], self.max_text_len),
                             float('-inf'),
                             device=res.device)
        new_res[..., :res.shape[-1]] = res

        return new_res


@MODELS.register_module()
class GroundingDINOHead(DINOHead):
    """Head of the Grounding DINO: Marrying DINO with Grounded Pre-Training for
    Open-Set Object Detection.

    Args:
        contrastive_cfg (dict, optional): Contrastive config that contains
          keys like ``max_text_len``. Defaults to dict(max_text_len=256).
    """

    def __init__(self, contrastive_cfg=dict(max_text_len=256), **kwargs):
        self.contrastive_cfg = contrastive_cfg
        self.max_text_len = contrastive_cfg.get('max_text_len', 256)
        super().__init__(**kwargs)

    def _init_layers(self) -> None:
        """Initialize classification branch and regression branch of head."""
        fc_cls = ContrastiveEmbed(**self.contrastive_cfg)
        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, 4))
        reg_branch = nn.Sequential(*reg_branch)

        # NOTE: due to the fc_cls is a contrastive embedding and don't
        # have any trainable parameters,we do not need to copy it.
        if self.share_pred_layer:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(self.num_pred_layer)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(self.num_pred_layer)])
        else:
            self.cls_branches = nn.ModuleList(
                [copy.deepcopy(fc_cls) for _ in range(self.num_pred_layer)])
            self.reg_branches = nn.ModuleList([
                copy.deepcopy(reg_branch) for _ in range(self.num_pred_layer)
            ])

    def init_weights(self) -> None:
        """Initialize weights of the Deformable DETR head."""
        for m in self.reg_branches:
            constant_init(m[-1], 0, bias=0)
        nn.init.constant_(self.reg_branches[0][-1].bias.data[2:], -2.0)
        if self.as_two_stage:
            for m in self.reg_branches:
                nn.init.constant_(m[-1].bias.data[2:], 0.0)

    def _get_targets_single(self, cls_score: Tensor, bbox_pred: Tensor,
                            gt_instances: InstanceData,
                            img_meta: dict) -> tuple:
        """Compute regression and classification targets for one image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_queries, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_queries, 4].
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It should includes ``bboxes`` and ``labels``
                attributes.
            img_meta (dict): Meta information for one image.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

            - labels (Tensor): Labels of each image.
            - label_weights (Tensor]): Label weights of each image.
            - bbox_targets (Tensor): BBox targets of each image.
            - bbox_weights (Tensor): BBox weights of each image.
            - pos_inds (Tensor): Sampled positive indices for each image.
            - neg_inds (Tensor): Sampled negative indices for each image.
        """
        img_h, img_w = img_meta['img_shape']
        factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)
        num_bboxes = bbox_pred.size(0)
        # convert bbox_pred from xywh, normalized to xyxy, unnormalized
        bbox_pred = bbox_cxcywh_to_xyxy(bbox_pred)
        bbox_pred = bbox_pred * factor

        pred_instances = InstanceData(scores=cls_score, bboxes=bbox_pred)
        # assigner and sampler
        assign_result = self.assigner.assign(
            pred_instances=pred_instances,
            gt_instances=gt_instances,
            img_meta=img_meta)
        gt_bboxes = gt_instances.bboxes

        pos_inds = torch.nonzero(
            assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(
            assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1
        pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds.long(), :]

        # Major changes. The labels are 0-1 binary labels for each bbox
        # and text tokens.
        labels = gt_bboxes.new_full((num_bboxes, self.max_text_len),
                                    0,
                                    dtype=torch.float32)
        labels[pos_inds] = gt_instances.positive_maps[pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred, dtype=gt_bboxes.dtype)
        bbox_weights = torch.zeros_like(bbox_pred, dtype=gt_bboxes.dtype)
        bbox_weights[pos_inds] = 1.0

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        pos_gt_bboxes_normalized = pos_gt_bboxes / factor
        pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)
        bbox_targets[pos_inds] = pos_gt_bboxes_targets
        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds)

    def forward(
        self,
        hidden_states: Tensor,
        references: List[Tensor],
        memory_text: Tensor,
        text_token_mask: Tensor,
    ) -> Tuple[Tensor]:
        """Forward function.

        Args:
            hidden_states (Tensor): Hidden states output from each decoder
                layer, has shape (num_decoder_layers, bs, num_queries, dim).
            references (List[Tensor]): List of the reference from the decoder.
                The first reference is the `init_reference` (initial) and the
                other num_decoder_layers(6) references are `inter_references`
                (intermediate). The `init_reference` has shape (bs,
                num_queries, 4) when `as_two_stage` of the detector is `True`,
                otherwise (bs, num_queries, 2). Each `inter_reference` has
                shape (bs, num_queries, 4) when `with_box_refine` of the
                detector is `True`, otherwise (bs, num_queries, 2). The
                coordinates are arranged as (cx, cy) when the last dimension is
                2, and (cx, cy, w, h) when it is 4.
            memory_text (Tensor): Memory text. It has shape (bs, len_text,
                text_embed_dims).
            text_token_mask (Tensor): Text token mask. It has shape (bs,
                len_text).

        Returns:
            tuple[Tensor]: results of head containing the following tensor.

            - all_layers_outputs_classes (Tensor): Outputs from the
              classification head, has shape (num_decoder_layers, bs,
              num_queries, cls_out_channels).
            - all_layers_outputs_coords (Tensor): Sigmoid outputs from the
              regression head with normalized coordinate format (cx, cy, w,
              h), has shape (num_decoder_layers, bs, num_queries, 4) with the
              last dimension arranged as (cx, cy, w, h).
        """
        all_layers_outputs_classes = []
        all_layers_outputs_coords = []
        self.bbox_weights = []
        for layer_id in range(hidden_states.shape[0]):
            reference = inverse_sigmoid(references[layer_id])
            # NOTE The last reference will not be used.
            hidden_state = hidden_states[layer_id] # (bs, 1100, 256)
            outputs_class = self.cls_branches[layer_id](hidden_state,
                                                        memory_text,
                                                        text_token_mask)
            tmp_reg_preds = self.reg_branches[layer_id](hidden_state)
            if reference.shape[-1] == 4:
                # When `layer` is 0 and `as_two_stage` of the detector
                # is `True`, or when `layer` is greater than 0 and
                # `with_box_refine` of the detector is `True`.
                tmp_reg_preds += reference
            else:
                # When `layer` is 0 and `as_two_stage` of the detector
                # is `False`, or when `layer` is greater than 0 and
                # `with_box_refine` of the detector is `False`.
                assert reference.shape[-1] == 2
                tmp_reg_preds[..., :2] += reference
            outputs_coord = tmp_reg_preds.sigmoid()
            all_layers_outputs_classes.append(outputs_class)
            all_layers_outputs_coords.append(outputs_coord)

        all_layers_outputs_classes = torch.stack(all_layers_outputs_classes)
        all_layers_outputs_coords = torch.stack(all_layers_outputs_coords)

        return all_layers_outputs_classes, all_layers_outputs_coords

    def predict(self,
                hidden_states: Tensor,
                references: List[Tensor],
                memory_text: Tensor,
                text_token_mask: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> InstanceList:
        """Perform forward propagation and loss calculation of the detection
        head on the queries of the upstream network.

        Args:
            hidden_states (Tensor): Hidden states output from each decoder
                layer, has shape (num_decoder_layers, num_queries, bs, dim).
            references (List[Tensor]): List of the reference from the decoder.
                The first reference is the `init_reference` (initial) and the
                other num_decoder_layers(6) references are `inter_references`
                (intermediate). The `init_reference` has shape (bs,
                num_queries, 4) when `as_two_stage` of the detector is `True`,
                otherwise (bs, num_queries, 2). Each `inter_reference` has
                shape (bs, num_queries, 4) when `with_box_refine` of the
                detector is `True`, otherwise (bs, num_queries, 2). The
                coordinates are arranged as (cx, cy) when the last dimension is
                2, and (cx, cy, w, h) when it is 4.
            memory_text (Tensor): Memory text. It has shape (bs, len_text,
                text_embed_dims).
            text_token_mask (Tensor): Text token mask. It has shape (bs,
                len_text).
            batch_data_samples (SampleList): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool, optional): If `True`, return boxes in original
                image space. Defaults to `True`.

        Returns:
            InstanceList: Detection results of each image
                after the post process.
        """
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        batch_token_positive_maps = [
            data_samples.token_positive_map
            for data_samples in batch_data_samples
        ]

        outs = self(hidden_states, references, memory_text, text_token_mask)

        predictions = self.predict_by_feat(
            *outs,
            batch_img_metas=batch_img_metas,
            batch_token_positive_maps=batch_token_positive_maps,
            rescale=rescale)
        return predictions

    def predict_by_feat(self,
                        all_layers_cls_scores: Tensor,
                        all_layers_bbox_preds: Tensor,
                        batch_img_metas: List[Dict],
                        batch_token_positive_maps: Optional[List[dict]] = None,
                        rescale: bool = False) -> InstanceList:
        """Transform a batch of output features extracted from the head into
        bbox results.

        Args:
            all_layers_cls_scores (Tensor):  Classification scores of all
                decoder layers, has shape (num_decoder_layers, bs, num_queries,
                cls_out_channels).
            all_layers_bbox_preds (Tensor): Regression outputs of all decoder
                layers. Each is a 4D-tensor with normalized coordinate format
                (cx, cy, w, h) and shape (num_decoder_layers, bs, num_queries,
                4) with the last dimension arranged as (cx, cy, w, h).
            batch_img_metas (List[Dict]): _description_
            batch_token_positive_maps (list[dict], Optional): Batch token
                positive map. Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Object detection results of each image
            after the post process. Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        cls_scores = all_layers_cls_scores[-1]
        bbox_preds = all_layers_bbox_preds[-1]
        result_list = []
        for img_id in range(len(batch_img_metas)):
            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]
            img_meta = batch_img_metas[img_id]
            token_positive_maps = batch_token_positive_maps[img_id]
            results = self._predict_by_feat_single(cls_score, bbox_pred,
                                                   token_positive_maps,
                                                   img_meta, rescale)
            result_list.append(results)
        return result_list

    def _predict_by_feat_single(self,
                                cls_score: Tensor,
                                bbox_pred: Tensor,
                                token_positive_maps: dict,
                                img_meta: dict,
                                rescale: bool = True) -> InstanceData:
        """Transform a single image's features extracted from the head into
        bbox results.

        Args:
            cls_score (Tensor): Box score logits from the last decoder layer
                for each image. Shape [num_queries, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from the last decoder layer
                for each image, with coordinate format (cx, cy, w, h) and
                shape [num_queries, 4].
            token_positive_maps (dict): Token positive map.
            img_meta (dict): Image meta info.
            rescale (bool, optional): If True, return boxes in original image
                space. Default True.

        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        assert len(cls_score) == len(bbox_pred)  # num_queries
        max_per_img = self.test_cfg.get('max_per_img', len(cls_score))
        img_shape = img_meta['img_shape']

        if token_positive_maps is not None:
            cls_score = convert_grounding_to_cls_scores(
                logits=cls_score.sigmoid()[None],
                positive_maps=[token_positive_maps])[0]
            scores, indexes = cls_score.view(-1).topk(max_per_img)
            num_classes = cls_score.shape[-1]
            det_labels = indexes % num_classes
            bbox_index = indexes // num_classes
            bbox_pred = bbox_pred[bbox_index]
        else:
            cls_score = cls_score.sigmoid()
            scores, _ = cls_score.max(-1)
            scores, indexes = scores.topk(max_per_img)
            bbox_pred = bbox_pred[indexes]
            det_labels = scores.new_zeros(scores.shape, dtype=torch.long)

        det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred)
        det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
        det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
        det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
        if rescale:
            assert img_meta.get('scale_factor') is not None
            det_bboxes /= det_bboxes.new_tensor(
                img_meta['scale_factor']).repeat((1, 2))
        results = InstanceData()
        results.bboxes = det_bboxes
        results.scores = scores
        results.labels = det_labels
        return results

    def loss(self, hidden_states: Tensor, references: List[Tensor],
             memory_text: Tensor, text_token_mask: Tensor,
             enc_outputs_class: Tensor, enc_outputs_coord: Tensor,
             batch_data_samples: SampleList, dn_meta: Dict[str, int]) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the queries of the upstream network.

        Args:
            hidden_states (Tensor): Hidden states output from each decoder
                layer, has shape (num_decoder_layers, bs, num_queries_total,
                dim), where `num_queries_total` is the sum of
                `num_denoising_queries` and `num_matching_queries` when
                `self.training` is `True`, else `num_matching_queries`.
            references (list[Tensor]): List of the reference from the decoder.
                The first reference is the `init_reference` (initial) and the
                other num_decoder_layers(6) references are `inter_references`
                (intermediate). The `init_reference` has shape (bs,
                num_queries_total, 4) and each `inter_reference` has shape
                (bs, num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h).
            memory_text (Tensor): Memory text. It has shape (bs, len_text,
                text_embed_dims).
            enc_outputs_class (Tensor): The score of each point on encode
                feature map, has shape (bs, num_feat_points, cls_out_channels).
            enc_outputs_coord (Tensor): The proposal generate from the
                encode feature map, has shape (bs, num_feat_points, 4) with the
                last dimension arranged as (cx, cy, w, h).
            batch_data_samples (list[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'. It will be used for split outputs of
              denoising and matching parts and loss calculation.

        Returns:
            dict: A dictionary of loss components.
        """
        batch_gt_instances = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)

        outs = self(hidden_states, references, memory_text, text_token_mask)
        self.text_masks = text_token_mask
        loss_inputs = outs + (enc_outputs_class, enc_outputs_coord,
                              batch_gt_instances, batch_img_metas, dn_meta)
        
        losses = self.loss_by_feat(*loss_inputs)
        return losses
   
    def loss_by_feat(
        self,
        all_layers_cls_scores: Tensor,
        all_layers_bbox_preds: Tensor,
        enc_cls_scores: Tensor,
        enc_bbox_preds: Tensor,
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        dn_meta: Dict[str, int],
        batch_gt_instances_ignore= None) -> Dict[str, Tensor]:
        """Loss function.
        Args:
            all_layers_cls_scores (Tensor): Classification scores of all
                decoder layers, has shape (num_decoder_layers, bs,
                num_queries_total, cls_out_channels), where
                `num_queries_total` is the sum of `num_denoising_queries`
                and `num_matching_queries`.
            all_layers_bbox_preds (Tensor): Regression outputs of all decoder
                layers. Each is a 4D-tensor with normalized coordinate format
                (cx, cy, w, h) and has shape (num_decoder_layers, bs,
                num_queries_total, 4).
            enc_cls_scores (Tensor): The score of each point on encode
                feature map, has shape (bs, num_feat_points, cls_out_channels).
            enc_bbox_preds (Tensor): The proposal generate from the encode
                feature map, has shape (bs, num_feat_points, 4) with the last
                dimension arranged as (cx, cy, w, h).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            dn_meta (Dict[str, int]): The dictionary saves information about
                group collation, including 'num_denoising_queries' and
                'num_denoising_groups'. It will be used for split outputs of
                denoising and matching parts and loss calculation.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # extract denoising and matching part of outputs
        (all_layers_matching_cls_scores, all_layers_matching_bbox_preds,
         all_layers_denoising_cls_scores, all_layers_denoising_bbox_preds) = \
            self.split_outputs(
                all_layers_cls_scores, all_layers_bbox_preds, dn_meta)

        loss_dict = super(DeformableDETRHead, self).loss_by_feat(
            all_layers_matching_cls_scores, all_layers_matching_bbox_preds,
            batch_gt_instances, batch_img_metas, batch_gt_instances_ignore)
        # NOTE DETRHead.loss_by_feat but not DeformableDETRHead.loss_by_feat
        # is called, because the encoder loss calculations are different
        # between DINO and DeformableDETR.

        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            # NOTE The enc_loss calculation of the DINO is
            # different from that of Deformable DETR.
            enc_loss_cls, enc_losses_bbox, enc_losses_iou = \
                self.loss_by_feat_single(
                    enc_cls_scores, enc_bbox_preds,
                    batch_gt_instances=batch_gt_instances,
                    batch_img_metas=batch_img_metas)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox
            loss_dict['enc_loss_iou'] = enc_losses_iou

        if all_layers_denoising_cls_scores is not None:
            # calculate denoising loss from all decoder layers
            dn_losses_cls, dn_losses_bbox, dn_losses_iou = self.loss_dn(
                all_layers_denoising_cls_scores,
                all_layers_denoising_bbox_preds,
                batch_gt_instances=batch_gt_instances,
                batch_img_metas=batch_img_metas,
                dn_meta=dn_meta)
            # collate denoising loss
            loss_dict['dn_loss_cls'] = dn_losses_cls[-1]
            loss_dict['dn_loss_bbox'] = dn_losses_bbox[-1]
            loss_dict['dn_loss_iou'] = dn_losses_iou[-1]
            for num_dec_layer, (loss_cls_i, loss_bbox_i, loss_iou_i) in \
                    enumerate(zip(dn_losses_cls[:-1], dn_losses_bbox[:-1],
                                  dn_losses_iou[:-1])):
                loss_dict[f'd{num_dec_layer}.dn_loss_cls'] = loss_cls_i
                loss_dict[f'd{num_dec_layer}.dn_loss_bbox'] = loss_bbox_i
                loss_dict[f'd{num_dec_layer}.dn_loss_iou'] = loss_iou_i
        return loss_dict
    
    def loss_by_feat_single(self, cls_scores: Tensor, bbox_preds: Tensor,
                            batch_gt_instances: InstanceList,
                            batch_img_metas: List[dict]) -> Tuple[Tensor]:
        """Loss function for outputs from a single decoder layer of a single
        feature level.

        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images, has shape (bs, num_queries, cls_out_channels).
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape (bs, num_queries, 4).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            Tuple[Tensor]: A tuple including `loss_cls`, `loss_box` and
            `loss_iou`.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        with torch.no_grad():
            cls_reg_targets = self.get_targets(cls_scores_list,
                                               bbox_preds_list,
                                               batch_gt_instances,
                                               batch_img_metas)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.stack(labels_list, 0)
        label_weights = torch.stack(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # ===== this change =====
        # Loss is not computed for the padded regions of the text.
        assert (self.text_masks.dim() == 2)
        text_masks = self.text_masks.new_zeros(
            (self.text_masks.size(0), self.max_text_len))
        text_masks[:, :self.text_masks.size(1)] = self.text_masks
        text_mask = (text_masks > 0).unsqueeze(1)
        text_mask = text_mask.repeat(1, cls_scores.size(1), 1)
        cls_scores = torch.masked_select(cls_scores, text_mask).contiguous()

        labels = torch.masked_select(labels, text_mask)
        label_weights = label_weights[...,
                                      None].repeat(1, 1, text_mask.size(-1))
        label_weights = torch.masked_select(label_weights, text_mask)

        # classification loss
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        if isinstance(self.loss_cls, QualityFocalLoss):
            raise NotImplementedError(
                'QualityFocalLoss for GroundingDINOHead is not supported yet.')
        else:
            loss_cls = self.loss_cls(
                cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(batch_img_metas, bbox_preds):
            img_h, img_w, = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)

        # regression L1 loss
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)
        
        self.bbox_weights.append(bbox_weights)
        return loss_cls, loss_bbox, loss_iou

    def _loss_dn_single(self, dn_cls_scores: Tensor, dn_bbox_preds: Tensor,
                        batch_gt_instances: InstanceList,
                        batch_img_metas: List[dict],
                        dn_meta: Dict[str, int]) -> Tuple[Tensor]:
        """Denoising loss for outputs from a single decoder layer.

        Args:
            dn_cls_scores (Tensor): Classification scores of a single decoder
                layer in denoising part, has shape (bs, num_denoising_queries,
                cls_out_channels).
            dn_bbox_preds (Tensor): Regression outputs of a single decoder
                layer in denoising part. Each is a 4D-tensor with normalized
                coordinate format (cx, cy, w, h) and has shape
                (bs, num_denoising_queries, 4).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'. It will be used for split outputs of
              denoising and matching parts and loss calculation.

        Returns:
            Tuple[Tensor]: A tuple including `loss_cls`, `loss_box` and
            `loss_iou`.
        """
        cls_reg_targets = self.get_dn_targets(batch_gt_instances,
                                              batch_img_metas, dn_meta)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.stack(labels_list, 0)
        label_weights = torch.stack(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        # ===== this change =====
        # Loss is not computed for the padded regions of the text.
        assert (self.text_masks.dim() == 2)
        text_masks = self.text_masks.new_zeros(
            (self.text_masks.size(0), self.max_text_len))
        text_masks[:, :self.text_masks.size(1)] = self.text_masks
        text_mask = (text_masks > 0).unsqueeze(1)
        text_mask = text_mask.repeat(1, dn_cls_scores.size(1), 1)
        cls_scores = torch.masked_select(dn_cls_scores, text_mask).contiguous()
        labels = torch.masked_select(labels, text_mask)
        label_weights = label_weights[...,
                                      None].repeat(1, 1, text_mask.size(-1))
        label_weights = torch.masked_select(label_weights, text_mask)
        # =======================

        # classification loss
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = \
            num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        if len(cls_scores) > 0:
            if isinstance(self.loss_cls, QualityFocalLoss):
                raise NotImplementedError('QualityFocalLoss is not supported')
            else:
                loss_cls = self.loss_cls(
                    cls_scores,
                    labels,
                    label_weights,
                    avg_factor=cls_avg_factor)
        else:
            loss_cls = torch.zeros(
                1, dtype=cls_scores.dtype, device=cls_scores.device)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(batch_img_metas, dn_bbox_preds):
            img_h, img_w = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = dn_bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)

        # regression L1 loss
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)
        return loss_cls, loss_bbox, loss_iou

    def _get_dn_targets_single(self, gt_instances: InstanceData,
                               img_meta: dict, dn_meta: Dict[str,
                                                             int]) -> tuple:
        """Get targets in denoising part for one image.

        Args:
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It should includes ``bboxes`` and ``labels``
                attributes.
            img_meta (dict): Meta information for one image.
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'. It will be used for split outputs of
              denoising and matching parts and loss calculation.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

            - labels (Tensor): Labels of each image.
            - label_weights (Tensor]): Label weights of each image.
            - bbox_targets (Tensor): BBox targets of each image.
            - bbox_weights (Tensor): BBox weights of each image.
            - pos_inds (Tensor): Sampled positive indices for each image.
            - neg_inds (Tensor): Sampled negative indices for each image.
        """
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        num_groups = dn_meta['num_denoising_groups']
        num_denoising_queries = dn_meta['num_denoising_queries']
        num_queries_each_group = int(num_denoising_queries / num_groups)
        device = gt_bboxes.device

        if len(gt_labels) > 0:
            t = torch.arange(len(gt_labels), dtype=torch.long, device=device)
            t = t.unsqueeze(0).repeat(num_groups, 1)
            pos_assigned_gt_inds = t.flatten()
            pos_inds = torch.arange(
                num_groups, dtype=torch.long, device=device)
            pos_inds = pos_inds.unsqueeze(1) * num_queries_each_group + t
            pos_inds = pos_inds.flatten()
        else:
            pos_inds = pos_assigned_gt_inds = \
                gt_bboxes.new_tensor([], dtype=torch.long)

        neg_inds = pos_inds + num_queries_each_group // 2
        # label targets
        # this change
        labels = gt_bboxes.new_full((num_denoising_queries, self.max_text_len),
                                    0,
                                    dtype=torch.float32)
        labels[pos_inds] = gt_instances.positive_maps[pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_denoising_queries)

        # bbox targets
        bbox_targets = torch.zeros(num_denoising_queries, 4, device=device)
        bbox_weights = torch.zeros(num_denoising_queries, 4, device=device)
        bbox_weights[pos_inds] = 1.0
        img_h, img_w = img_meta['img_shape']

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        factor = gt_bboxes.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)
        gt_bboxes_normalized = gt_bboxes / factor
        gt_bboxes_targets = bbox_xyxy_to_cxcywh(gt_bboxes_normalized)
        bbox_targets[pos_inds] = gt_bboxes_targets.repeat([num_groups, 1])

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds)

def weighting_function(reg_max, up, reg_scale, deploy=False):
    """
    Generates the non-uniform Weighting Function W(n) for bounding box regression.

    Args:
        reg_max (int): Max number of the discrete bins.
        up (Tensor): Controls upper bounds of the sequence,
                     where maximum offset is ±up * H / W.
        reg_scale (float): Controls the curvature of the Weighting Function.
                           Larger values result in flatter weights near the central axis W(reg_max/2)=0
                           and steeper weights at both ends.
        deploy (bool): If True, uses deployment mode settings.

    Returns:
        Tensor: Sequence of Weighting Function.
    """
    if deploy:
        upper_bound1 = (abs(up[0]) * abs(reg_scale)).item()
        upper_bound2 = (abs(up[0]) * abs(reg_scale) * 2).item()
        step = (upper_bound1 + 1) ** (2 / (reg_max - 2))
        left_values = [-((step) ** i) + 1 for i in range(reg_max // 2 - 1, 0, -1)]
        right_values = [(step) ** i - 1 for i in range(1, reg_max // 2)]
        values = (
            [-upper_bound2]
            + left_values
            + [torch.zeros_like(up[0][None])]
            + right_values
            + [upper_bound2]
        )
        return torch.tensor(values, dtype=up.dtype, device=up.device)
    else:
        upper_bound1 = abs(up[0]) * abs(reg_scale)
        upper_bound2 = abs(up[0]) * abs(reg_scale) * 2
        step = (upper_bound1 + 1) ** (2 / (reg_max - 2))
        left_values = [-((step) ** i) + 1 for i in range(reg_max // 2 - 1, 0, -1)]
        right_values = [(step) ** i - 1 for i in range(1, reg_max // 2)]
        values = (
            [-upper_bound2]
            + left_values
            + [torch.zeros_like(up[0][None])]
            + right_values
            + [upper_bound2]
        )
        return torch.cat(values, 0)
def translate_gt(gt, reg_max, reg_scale, up):
    """
    Decodes bounding box ground truth (GT) values into distribution-based GT representations.

    This function maps continuous GT values into discrete distribution bins, which can be used
    for regression tasks in object detection models. It calculates the indices of the closest
    bins to each GT value and assigns interpolation weights to these bins based on their proximity
    to the GT value.

    Args:
        gt (Tensor): Ground truth bounding box values, shape (N, ).
        reg_max (int): Maximum number of discrete bins for the distribution.
        reg_scale (float): Controls the curvature of the Weighting Function.
        up (Tensor): Controls the upper bounds of the Weighting Function.

    Returns:
        Tuple[Tensor, Tensor, Tensor]:
            - indices (Tensor): Index of the left bin closest to each GT value, shape (N, ).
            - weight_right (Tensor): Weight assigned to the right bin, shape (N, ).
            - weight_left (Tensor): Weight assigned to the left bin, shape (N, ).
    """
    gt = gt.reshape(-1)
    function_values = weighting_function(reg_max, up, reg_scale)

    # Find the closest left-side indices for each value
    diffs = function_values.unsqueeze(0) - gt.unsqueeze(1)
    mask = diffs <= 0
    closest_left_indices = torch.sum(mask, dim=1) - 1

    # Calculate the weights for the interpolation
    indices = closest_left_indices.float()

    weight_right = torch.zeros_like(indices)
    weight_left = torch.zeros_like(indices)

    valid_idx_mask = (indices >= 0) & (indices < reg_max)
    valid_indices = indices[valid_idx_mask].long()

    # Obtain distances
    left_values = function_values[valid_indices]
    right_values = function_values[valid_indices + 1]

    left_diffs = torch.abs(gt[valid_idx_mask] - left_values)
    right_diffs = torch.abs(right_values - gt[valid_idx_mask])

    # Valid weights
    weight_right[valid_idx_mask] = left_diffs / (left_diffs + right_diffs)
    weight_left[valid_idx_mask] = 1.0 - weight_right[valid_idx_mask]

    # Invalid weights (out of range)
    invalid_idx_mask_neg = indices < 0
    weight_right[invalid_idx_mask_neg] = 0.0
    weight_left[invalid_idx_mask_neg] = 1.0
    indices[invalid_idx_mask_neg] = 0.0

    invalid_idx_mask_pos = indices >= reg_max
    weight_right[invalid_idx_mask_pos] = 1.0
    weight_left[invalid_idx_mask_pos] = 0.0
    indices[invalid_idx_mask_pos] = reg_max - 0.1

    return indices, weight_right, weight_left
from mmdet.models.layers.transformer.grounding_dino_layers import distance2bbox
def bbox2distance(points, bbox, reg_max, reg_scale, up, eps=0.1):
    """
    Converts bounding box coordinates to distances from a reference point.

    Args:
        points (Tensor): (n, 4) [x, y, w, h], where (x, y) is the center.
        bbox (Tensor): (n, 4) bounding boxes in "xyxy" format.
        reg_max (float): Maximum bin value.
        reg_scale (float): Controling curvarture of W(n).
        up (Tensor): Controling upper bounds of W(n).
        eps (float): Small value to ensure target < reg_max.

    Returns:
        Tensor: Decoded distances.
    """
    reg_scale = abs(reg_scale)
    left = (points[:, 0] - bbox[:, 0]) / (points[..., 2] / reg_scale + 1e-16) - 0.5 * reg_scale
    top = (points[:, 1] - bbox[:, 1]) / (points[..., 3] / reg_scale + 1e-16) - 0.5 * reg_scale
    right = (bbox[:, 2] - points[:, 0]) / (points[..., 2] / reg_scale + 1e-16) - 0.5 * reg_scale
    bottom = (bbox[:, 3] - points[:, 1]) / (points[..., 3] / reg_scale + 1e-16) - 0.5 * reg_scale
    four_lens = torch.stack([left, top, right, bottom], -1)
    four_lens, weight_right, weight_left = translate_gt(four_lens, reg_max, reg_scale, up)
    if reg_max is not None:
        four_lens = four_lens.clamp(min=0, max=reg_max - eps)
    return four_lens.reshape(-1).detach(), weight_right.detach(), weight_left.detach()

# from mmengine.model import BaseModule, ModuleList
# from mmdet.models.layers.transformer.utils import MLP
from mmdet.utils import ConfigType,OptInstanceList
import torch.nn.functional as F
from ..utils import multi_apply


class Integral(nn.Module):
    def __init__(self, reg_max=32, up=torch.tensor([0.5]), reg_scale=torch.tensor([4.0])):
        super(Integral, self).__init__()
        self.reg_max = reg_max
        self.reg_scale = reg_scale
        self.up = up
        self.register_buffer('project', weighting_function(self.reg_max, up, reg_scale))
    def forward(self, x):
        shape = x.shape
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = F.linear(x, self.project.to(x.device)).reshape(-1, 4)
        return x.reshape(list(shape[:-1]) + [-1])

@MODELS.register_module()
class GroundingDFINEHead(GroundingDINOHead):
    def __init__(self,
                 reg_max:int=32,
                 loss_dfl: ConfigType = dict(type='DecoupledDistributionFocalLoss', loss_weight=0.25),
                 loss_ld: ConfigType = dict(type='KnowledgeDistillationKLDivLoss', loss_weight=0.25, T=5),
                 **kwargs):
        self.reg_max = reg_max
        super().__init__(**kwargs)
        
        self.up = nn.Parameter(torch.tensor([0.5]), requires_grad=False)
        self.reg_scale = nn.Parameter(torch.tensor([4.0]), requires_grad=False)
        # self.project = weighting_function(self.reg_max, self.up, self.reg_scale)
        self.integral = Integral(self.reg_max, self.up, self.reg_scale)
        
        #location quality estimation
        # self.lqe_branches = nn.ModuleList(
        #     [LQE(4, 64, 2, self.reg_max) for _ in range(self.num_pred_layer)]
        # )
        self.loss_dfl = MODELS.build(loss_dfl)
        self.loss_ld = MODELS.build(loss_ld)
    
    def _init_layers(self) -> None:
        """Initialize classification branch and regression branch of head."""
        fc_cls = ContrastiveEmbed(**self.contrastive_cfg)
        def mlp(in_dims, hidden_dims, out_dims, layers):
            mlp = []
            for i in range(layers-1):
                mlp.append(Linear(in_dims, hidden_dims))
                mlp.append(nn.ReLU())
                in_dims = hidden_dims
            mlp.append(Linear(hidden_dims, out_dims))
            return nn.Sequential(*mlp)
        reg_branch = mlp(self.embed_dims, self.embed_dims, 4 * (self.reg_max + 1), self.num_reg_fcs+1)
        bbox_branch = mlp(self.embed_dims, self.embed_dims, 4, self.num_reg_fcs+1)
        
        # NOTE: due to the fc_cls is a contrastive embedding and don't
        # have any trainable parameters,we do not need to copy it.
        if self.share_pred_layer:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(self.num_pred_layer)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(self.num_pred_layer-1)])
            self.reg_branches.append(bbox_branch)
        else:
            self.cls_branches = nn.ModuleList(
                [copy.deepcopy(fc_cls) for _ in range(self.num_pred_layer)])
            self.reg_branches = nn.ModuleList([
                copy.deepcopy(reg_branch) for _ in range(self.num_pred_layer)])
            self.reg_branches[-1] = bbox_branch

    def forward(self, hidden_states: Tensor,
                    references: List[Tensor],
                    memory_text: Tensor,
                    text_token_mask: Tensor) -> Tuple[Tensor]:
        all_layers_outputs_classes = []
        all_layers_outputs_corners = []
        all_layers_outputs_coords = []
        self.ref_points = torch.stack(references,dim=0)
        
        reference = references[-1].detach()
        for layer_id in range(hidden_states.shape[0]):
            # reference = references[layer_id]
            # NOTE The last reference will not be used.
            hidden_state = hidden_states[layer_id]
            outputs_class = self.cls_branches[layer_id](hidden_state,
                                                        memory_text,
                                                        text_token_mask)
            tmp_reg_corners = self.reg_branches[layer_id](hidden_state) #(bs, num_queries, 4 * (self.reg_max + 1))
            outputs_coord = distance2bbox(reference, self.integral(tmp_reg_corners), self.integral.reg_scale)
            
            reference = outputs_coord.detach()
            all_layers_outputs_classes.append(outputs_class)
            all_layers_outputs_coords.append(outputs_coord)
            all_layers_outputs_corners.append(tmp_reg_corners)
            
        all_layers_outputs_classes = torch.stack(all_layers_outputs_classes)
        all_layers_outputs_coords = torch.stack(all_layers_outputs_coords)
        all_layers_outputs_corners = torch.stack(all_layers_outputs_corners)
        if self.training:
            return all_layers_outputs_classes, all_layers_outputs_coords, all_layers_outputs_corners
        else:
            return all_layers_outputs_classes, all_layers_outputs_coords   

    def loss_by_feat(self,
                    all_layers_cls_scores: Tensor,
                    all_layers_bbox_preds: Tensor,
                    all_layers_corner_preds: Tensor,
                    enc_cls_scores: Tensor,
                    enc_bbox_preds: Tensor,
                    batch_gt_instances: InstanceList,
                    batch_img_metas: List[dict],
                    dn_meta: Dict[str, int],
                    batch_gt_instances_ignore: OptInstanceList = None) -> Dict[str, Tensor]:
        """Loss function.

        Args:
            all_layers_cls_scores (Tensor): Classification scores of all
                decoder layers, has shape (num_decoder_layers, bs,
                num_queries_total, cls_out_channels), where
                `num_queries_total` is the sum of `num_denoising_queries`
                and `num_matching_queries`.
            all_layers_bbox_preds (Tensor): Regression outputs of all decoder
                layers. Each is a 4D-tensor with normalized coordinate format
                (cx, cy, w, h) and has shape (num_decoder_layers, bs,
                num_queries_total, 4).
            enc_cls_scores (Tensor): The score of each point on encode
                feature map, has shape (bs, num_feat_points, cls_out_channels).
            enc_bbox_preds (Tensor): The proposal generate from the encode
                feature map, has shape (bs, num_feat_points, 4) with the last
                dimension arranged as (cx, cy, w, h).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            dn_meta (Dict[str, int]): The dictionary saves information about
                group collation, including 'num_denoising_queries' and
                'num_denoising_groups'. It will be used for split outputs of
                denoising and matching parts and loss calculation.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        num_layers = all_layers_corner_preds.shape[0]
        all_layers_soft_corner_preds = all_layers_corner_preds[-1].repeat(num_layers, 1, 1, 1)
        
        # extract denoising and matching part of outputs (num_layers, bs, num_queries, classes/bbox/regmax+1)
        all_layers_matching_cls_scores,all_layers_denoising_cls_scores = self.split_outputs(all_layers_cls_scores, dn_meta)
        all_layers_matching_bbox_preds,all_layers_denoising_bbox_preds = self.split_outputs(all_layers_bbox_preds, dn_meta)
        all_layers_matching_corner_preds,all_layers_denoising_corner_preds = self.split_outputs(all_layers_corner_preds, dn_meta)
        all_layers_matching_ref_points,all_layers_denoising_ref_points = self.split_outputs(self.ref_points, dn_meta)
        all_layers_matching_soft_corner_preds,all_layers_denoising_soft_corner_preds = self.split_outputs(all_layers_soft_corner_preds, dn_meta)

        assert batch_gt_instances_ignore is None, f'{self.__class__.__name__} only supports for batch_gt_instances_ignore setting to None.'
        losses_cls, losses_bbox, losses_iou = multi_apply(
            self.loss_by_feat_single,
            all_layers_matching_cls_scores,
            all_layers_matching_bbox_preds,
            batch_gt_instances=batch_gt_instances,
            batch_img_metas=batch_img_metas)
        losses_dfl, losses_ld = multi_apply(
            self.specific_loss_by_feat_single,
            all_layers_matching_cls_scores,
            all_layers_matching_bbox_preds,
            all_layers_matching_corner_preds,
            all_layers_matching_ref_points,
            all_layers_matching_soft_corner_preds,
            batch_gt_instances=batch_gt_instances,
            batch_img_metas=batch_img_metas)
        
        loss_dict = dict()
        def add_decoder_layer_loss(loss_dict, losses, t):
            num_dec_layer = 0
            # loss from the last decoder layer
            loss_dict[t]=losses[-1]
            for loss in losses[:-1]:
                loss_dict[f'd{num_dec_layer}.{t}'] = loss
                num_dec_layer += 1
            return loss_dict
        loss_dict = add_decoder_layer_loss(loss_dict, losses_cls, 'loss_cls')
        loss_dict = add_decoder_layer_loss(loss_dict, losses_bbox, 'loss_bbox')
        loss_dict = add_decoder_layer_loss(loss_dict, losses_iou, 'loss_iou')
        loss_dict = add_decoder_layer_loss(loss_dict, losses_dfl, 'loss_dfl')
        loss_dict = add_decoder_layer_loss(loss_dict, losses_ld, 'loss_ld')

        if enc_cls_scores is not None:
            # NOTE The enc_loss calculation of the DINO is
            # different from that of Deformable DETR.
            enc_loss_cls, enc_losses_bbox, enc_losses_iou = \
                self.loss_by_feat_single(
                    enc_cls_scores, enc_bbox_preds,
                    batch_gt_instances=batch_gt_instances,
                    batch_img_metas=batch_img_metas)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox
            loss_dict['enc_loss_iou'] = enc_losses_iou

        if all_layers_denoising_cls_scores is not None:
            # calculate denoising loss from all decoder layers
            dn_losses_cls, dn_losses_bbox, dn_losses_iou, dn_losses_dfl = self.loss_dn(
                all_layers_denoising_cls_scores,
                all_layers_denoising_bbox_preds,
                all_layers_denoising_corner_preds,
                all_layers_denoising_ref_points,
                batch_gt_instances=batch_gt_instances,
                batch_img_metas=batch_img_metas,
                dn_meta=dn_meta)
            # collate denoising loss
            loss_dict = add_decoder_layer_loss(loss_dict, dn_losses_cls, 'dn_loss_cls')
            loss_dict = add_decoder_layer_loss(loss_dict, dn_losses_bbox, 'dn_loss_bbox')
            loss_dict = add_decoder_layer_loss(loss_dict, dn_losses_iou, 'dn_loss_iou')
            loss_dict = add_decoder_layer_loss(loss_dict, dn_losses_dfl, 'dn_loss_dfl')
            # loss_dict = add_decoder_layer_loss(loss_dict, losses_ld, 'loss_ld')
        return loss_dict
    
    def specific_loss_by_feat_single(self, 
                                    cls_scores: Tensor, bbox_preds: Tensor,
                                    corner_preds: Tensor, ref_points: Tensor,
                                    soft_corners: Tensor,
                                    batch_gt_instances: InstanceList,
                                    batch_img_metas: List[dict]) -> Tuple[Tensor]:
        
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           batch_gt_instances, batch_img_metas)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        
        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(batch_img_metas, bbox_preds):
            img_h, img_w, = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets)
        pred_corners = corner_preds.reshape(-1, self.reg_max + 1)
        target_corners, weight_left, weight_right = bbox2distance(ref_points.reshape(-1,4), 
                                                                  bboxes_gt, 
                                                                  self.reg_max, self.reg_scale, self.up)
                
        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total = num_total_pos + num_total_neg
        num_total = max(num_total, 1)
        # dfl loss
        loss_dfl = self.loss_dfl(
                pred_corners,
                target_corners,
                weight_left, weight_right,
                weight=bbox_weights.reshape(-1),
                avg_factor=num_total)
        
        soft_corners = soft_corners.reshape(-1, self.reg_max + 1)
        loss_ld_pos = self.loss_ld(
                pred_corners,#(N*4, (self.reg_max + 1))
                soft_corners,
                weight=bbox_weights.reshape(-1),
                avg_factor=None,
                reduction_override='sum')
        loss_ld_neg = self.loss_ld(
                pred_corners,#(N*4, (self.reg_max + 1))
                soft_corners,
                weight=(1-bbox_weights).reshape(-1),
                avg_factor=None,
                reduction_override='sum')
        loss_ld = (loss_ld_neg * num_total_neg + loss_ld_pos * num_total_pos)/(num_total)
        return loss_dfl, loss_ld
        

    def loss_dn(self, all_layers_denoising_cls_scores: Tensor,
                all_layers_denoising_bbox_preds: Tensor,
                all_layers_denoising_corner_preds: Tensor,
                all_layers_denosing_ref_points: Tensor,
                batch_gt_instances: InstanceList, batch_img_metas: List[dict],
                dn_meta: Dict[str, int]) -> Tuple[List[Tensor]]:
        """Calculate denoising loss.

        Args:
            all_layers_denoising_cls_scores (Tensor): Classification scores of
                all decoder layers in denoising part, has shape (
                num_decoder_layers, bs, num_denoising_queries,
                cls_out_channels).
            all_layers_denoising_bbox_preds (Tensor): Regression outputs of all
                decoder layers in denoising part. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and has shape
                (num_decoder_layers, bs, num_denoising_queries, 4).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'. It will be used for split outputs of
              denoising and matching parts and loss calculation.

        Returns:
            Tuple[List[Tensor]]: The loss_dn_cls, loss_dn_bbox, and loss_dn_iou
            of each decoder layers.
        """
        return multi_apply(
            self._loss_dn_single,
            all_layers_denoising_cls_scores,
            all_layers_denoising_bbox_preds,
            all_layers_denoising_corner_preds,
            all_layers_denosing_ref_points,
            batch_gt_instances=batch_gt_instances,
            batch_img_metas=batch_img_metas,
            dn_meta=dn_meta)
    def _loss_dn_single(self, dn_cls_scores: Tensor, dn_bbox_preds: Tensor,
                        dn_corner_preds: Tensor, dn_ref_points: Tensor,
                        batch_gt_instances: InstanceList,
                        batch_img_metas: List[dict],
                        dn_meta: Dict[str, int]) -> Tuple[Tensor]:
        """Denoising loss for outputs from a single decoder layer.

        Args:
            dn_cls_scores (Tensor): Classification scores of a single decoder
                layer in denoising part, has shape (bs, num_denoising_queries,
                cls_out_channels).
            dn_bbox_preds (Tensor): Regression outputs of a single decoder
                layer in denoising part. Each is a 4D-tensor with normalized
                coordinate format (cx, cy, w, h) and has shape
                (bs, num_denoising_queries, 4).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'. It will be used for split outputs of
              denoising and matching parts and loss calculation.

        Returns:
            Tuple[Tensor]: A tuple including `loss_cls`, `loss_box` and
            `loss_iou`.
        """
        cls_reg_targets = self.get_dn_targets(batch_gt_instances,
                                              batch_img_metas, dn_meta)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.stack(labels_list, 0)
        label_weights = torch.stack(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        # ===== this change =====
        # Loss is not computed for the padded regions of the text.
        assert (self.text_masks.dim() == 2)
        text_masks = self.text_masks.new_zeros(
            (self.text_masks.size(0), self.max_text_len))
        text_masks[:, :self.text_masks.size(1)] = self.text_masks
        text_mask = (text_masks > 0).unsqueeze(1)
        text_mask = text_mask.repeat(1, dn_cls_scores.size(1), 1)
        cls_scores = torch.masked_select(dn_cls_scores, text_mask).contiguous()
        labels = torch.masked_select(labels, text_mask)
        label_weights = label_weights[...,
                                      None].repeat(1, 1, text_mask.size(-1))
        label_weights = torch.masked_select(label_weights, text_mask)
        # =======================

        # classification loss
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = \
            num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        if len(cls_scores) > 0:
            if isinstance(self.loss_cls, QualityFocalLoss):
                raise NotImplementedError('QualityFocalLoss is not supported')
            else:
                loss_cls = self.loss_cls(
                    cls_scores,
                    labels,
                    label_weights,
                    avg_factor=cls_avg_factor)
        else:
            loss_cls = torch.zeros(
                1, dtype=cls_scores.dtype, device=cls_scores.device)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(batch_img_metas, dn_bbox_preds):
            img_h, img_w = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = dn_bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)

        # regression L1 loss
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)
        
        pred_corners = dn_corner_preds.reshape(-1, self.reg_max + 1)
        
        target_corners, weight_left, weight_right = bbox2distance(dn_ref_points.reshape(-1, 4), 
                                                                  bboxes_gt/factors, 
                                                                  self.reg_max, self.reg_scale, self.up)
        # dfl loss
        num_total = num_total_pos + num_total_neg
        num_total = max(num_total, 1)
        loss_dfl = self.loss_dfl(
                pred_corners,
                target_corners,
                weight_left, weight_right,
                weight=bbox_weights.reshape(-1),
                avg_factor=num_total)
        
        return loss_cls, loss_bbox, loss_iou, loss_dfl
   
    @staticmethod
    def split_outputs(preds: Tensor,#(num_decoder_layers, bs, num_queries_total, n).
                      dn_meta: Dict[str, int]) -> Tuple[Tensor]:
        num_denoising_queries = dn_meta['num_denoising_queries']
        if dn_meta is not None:
            denoising_preds = preds[:, :, : num_denoising_queries, :]
            matching_preds = preds[:, :, num_denoising_queries:, :]
        else:
            denoising_preds = None
            matching_preds = preds
        return (matching_preds,denoising_preds)
    
    


@MODELS.register_module()
class HeadOptimizer(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, head, hidden_states: Tensor, references: List[Tensor],
                memory_text: Tensor, text_token_mask: Tensor,
                enc_outputs_class: Tensor, enc_outputs_coord: Tensor,
                batch_data_samples: SampleList, dn_meta: Dict[str, int]):
        
        # all_layers_outputs_classes, all_layers_outputs_coords, all_layers_outputs_corners = \
        #                                 head(hidden_states, references, memory_text, text_token_mask)
        batch_gt_instances = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)

        outs = head(hidden_states, references, memory_text, text_token_mask)
        all_layers_outputs_classes, all_layers_outputs_coords = outs
        head.text_masks = text_token_mask
        loss_inputs = outs + (enc_outputs_class, enc_outputs_coord,
                              batch_gt_instances, batch_img_metas, dn_meta)
        losses = head.loss_by_feat(*loss_inputs)
        """compute reward"""
        # giou最近1,最远-1
        loss_giou = [losses[f'd{i}.loss_iou'] for i in range(len(hidden_states)-1)]
        loss_giou.append(losses["loss_iou"])
        iou_reward = list(map(lambda x: (-x/head.loss_iou.loss_weight + 2), loss_giou)) 
        iou_reward = torch.stack(iou_reward, dim=-1)*head.loss_iou.loss_weight
        #
        loss_l1 = [losses[f'd{i}.loss_bbox'] for i in range(len(hidden_states)-1)]
        loss_l1.append(losses["loss_bbox"])
        loss_l1 = torch.stack(loss_l1, dim=-1)
        rewards = iou_reward-loss_l1
        advantages: Tensor = (rewards - rewards.mean(dim=-1)) / (rewards.std(-1) + 1e-6) # [num_layers, 1]
        
        """compute loss"""
        all_layers_matching_coords, _ = GroundingDFINEHead.split_outputs(all_layers_outputs_coords, dn_meta)
        bbox_weights = torch.stack(head.bbox_weights[:-1], 
                                   dim=0).reshape(*all_layers_matching_coords.shape[:-1], head.bbox_weights[0].shape[-1])
        loss = -torch.where(bbox_weights==1, all_layers_matching_coords, torch.zeros_like(all_layers_matching_coords)) #(num_layers, bs, num_queries, 4)
        loss = loss*advantages.reshape(-1, 1, 1, 1)
        return loss[bbox_weights==1].mean(-1)
        