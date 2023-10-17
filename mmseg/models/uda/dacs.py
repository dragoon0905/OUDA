# Obtained from: https://github.com/lhoyer/DAFormer
# Modifications:
# - Delete tensors after usage to free GPU memory
# - Add HRDA debug visualizations
# - Support ImageNet feature distance for LR and HR predictions of HRDA
# - Add masked image consistency
# - Update debug image system
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# The ema model update and the domain-mixing are based on:
# https://github.com/vikolss/DACS
# Copyright (c) 2020 vikolss. Licensed under the MIT License.
# A copy of the license is available at resources/license_dacs

import math
import os
import random
from copy import deepcopy

import mmcv
import numpy as np
import torch
from matplotlib import pyplot as plt
from timm.models.layers import DropPath
from torch.nn import functional as F
from torch.nn.modules.dropout import _DropoutNd

from mmseg.core import add_prefix
from mmseg.models import UDA, HRDAEncoderDecoder, build_segmentor
from mmseg.models.segmentors.hrda_encoder_decoder import crop
from mmseg.models.uda.masking_consistency_module import \
    MaskingConsistencyModule
from mmseg.models.uda.uda_decorator import UDADecorator, get_module
from mmseg.models.utils.dacs_transforms import (denorm, get_class_masks,
                                                get_mean_std, strong_transform, get_class_masks_resize)
from mmseg.models.utils.visualization import prepare_debug_out, subplotimg
from mmseg.utils.utils import downscale_label_ratio

# modified
from MobileSam.mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import cv2
from mmseg.ops.wrappers import resize


def _params_equal(ema_model, model):
    for ema_param, param in zip(ema_model.named_parameters(),
                                model.named_parameters()):
        if not torch.equal(ema_param[1].data, param[1].data):
            # print("Difference in", ema_param[0])
            return False
    return True


def calc_grad_magnitude(grads, norm_type=2.0):
    norm_type = float(norm_type)
    if norm_type == math.inf:
        norm = max(p.abs().max() for p in grads)
    else:
        norm = torch.norm(
            torch.stack([torch.norm(p, norm_type) for p in grads]), norm_type)

    return norm


@UDA.register_module()
class DACS(UDADecorator):

    def __init__(self, **cfg):
        super(DACS, self).__init__(**cfg)
        self.local_iter = 0
        self.max_iters = cfg['max_iters']
        self.source_only = cfg['source_only']
        self.alpha = cfg['alpha']
        self.pseudo_threshold = cfg['pseudo_threshold']
        self.psweight_ignore_top = cfg['pseudo_weight_ignore_top']
        self.psweight_ignore_bottom = cfg['pseudo_weight_ignore_bottom']
        self.fdist_lambda = cfg['imnet_feature_dist_lambda']
        self.fdist_classes = cfg['imnet_feature_dist_classes']
        self.fdist_scale_min_ratio = cfg['imnet_feature_dist_scale_min_ratio']
        self.enable_fdist = self.fdist_lambda > 0
        self.mix = cfg['mix']
        self.blur = cfg['blur']
        self.color_jitter_s = cfg['color_jitter_strength']
        self.color_jitter_p = cfg['color_jitter_probability']
        self.mask_mode = cfg['mask_mode']
        self.enable_masking = self.mask_mode is not None
        self.print_grad_magnitude = cfg['print_grad_magnitude']
        assert self.mix == 'class'

        self.debug_fdist_mask = None
        self.debug_gt_rescale = None

        self.class_probs = {}
        ema_cfg = deepcopy(cfg['model'])
        if not self.source_only:
            self.ema_model = build_segmentor(ema_cfg)
        self.mic = None
        if self.enable_masking:
            self.mic = MaskingConsistencyModule(require_teacher=False, cfg=cfg)
        if self.enable_fdist:
            self.imnet_model = build_segmentor(deepcopy(cfg['model']))
        else:
            self.imnet_model = None

        # modified
        self.is_dilation = cfg['is_dilation']
        self.is_bimix = cfg['is_bimix']
        self.is_SAM = cfg['is_SAM']
        self.is_shape = cfg['is_shape']
        self.SAM_ratio = cfg['SAM_ratio']
        
        model_type = "vit_t"
        sam_checkpoint = "./weights/mobile_sam.pt"

        device = "cuda" if torch.cuda.is_available() else "cpu"

        mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        mobile_sam.to(device=device)
        mobile_sam.eval()

        self.mask_generator = SamAutomaticMaskGenerator(mobile_sam)

    def get_ema_model(self):
        return get_module(self.ema_model)

    def get_imnet_model(self):
        return get_module(self.imnet_model)

    def _init_ema_weights(self):
        if self.source_only:
            return
        for param in self.get_ema_model().parameters():
            param.detach_()
        mp = list(self.get_model().parameters())
        mcp = list(self.get_ema_model().parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    def _update_ema(self, iter):
        if self.source_only:
            return
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        for ema_param, param in zip(self.get_ema_model().parameters(),
                                    self.get_model().parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """

        optimizer.zero_grad()
        log_vars = self(**data_batch)
        optimizer.step()

        log_vars.pop('loss', None)  # remove the unnecessary 'loss'
        outputs = dict(
            log_vars=log_vars, num_samples=len(data_batch['img_metas']))
        return outputs

    def masked_feat_dist(self, f1, f2, mask=None):
        feat_diff = f1 - f2
        # mmcv.print_log(f'fdiff: {feat_diff.shape}', 'mmseg')
        pw_feat_dist = torch.norm(feat_diff, dim=1, p=2)
        # mmcv.print_log(f'pw_fdist: {pw_feat_dist.shape}', 'mmseg')
        if mask is not None:
            # mmcv.print_log(f'fd mask: {mask.shape}', 'mmseg')
            pw_feat_dist = pw_feat_dist[mask.squeeze(1)]
            # mmcv.print_log(f'fd masked: {pw_feat_dist.shape}', 'mmseg')
        # If the mask is empty, the mean will be NaN. However, as there is
        # no connection in the compute graph to the network weights, the
        # network gradients are zero and no weight update will happen.
        # This can be verified with print_grad_magnitude.
        return torch.mean(pw_feat_dist)

    def calc_feat_dist(self, img, gt, feat=None):
        assert self.enable_fdist
        # Features from multiple input scales (see HRDAEncoderDecoder)
        if isinstance(self.get_model(), HRDAEncoderDecoder) and \
                self.get_model().feature_scale in \
                self.get_model().feature_scale_all_strs:
            lay = -1
            feat = [f[lay] for f in feat]
            with torch.no_grad():
                self.get_imnet_model().eval()
                feat_imnet = self.get_imnet_model().extract_feat(img)
                feat_imnet = [f[lay].detach() for f in feat_imnet]
            feat_dist = 0
            n_feat_nonzero = 0
            for s in range(len(feat_imnet)):
                if self.fdist_classes is not None:
                    fdclasses = torch.tensor(
                        self.fdist_classes, device=gt.device)
                    gt_rescaled = gt.clone()
                    if s in HRDAEncoderDecoder.last_train_crop_box:
                        gt_rescaled = crop(
                            gt_rescaled,
                            HRDAEncoderDecoder.last_train_crop_box[s])
                    scale_factor = gt_rescaled.shape[-1] // feat[s].shape[-1]
                    gt_rescaled = downscale_label_ratio(
                        gt_rescaled, scale_factor, self.fdist_scale_min_ratio,
                        self.num_classes, 255).long().detach()
                    fdist_mask = torch.any(gt_rescaled[..., None] == fdclasses,
                                           -1)
                    fd_s = self.masked_feat_dist(feat[s], feat_imnet[s],
                                                 fdist_mask)
                    feat_dist += fd_s
                    if fd_s != 0:
                        n_feat_nonzero += 1
                    del fd_s
                    if s == 0:
                        self.debug_fdist_mask = fdist_mask
                        self.debug_gt_rescale = gt_rescaled
                else:
                    raise NotImplementedError
        else:
            with torch.no_grad():
                self.get_imnet_model().eval()
                feat_imnet = self.get_imnet_model().extract_feat(img)
                feat_imnet = [f.detach() for f in feat_imnet]
            lay = -1
            if self.fdist_classes is not None:
                fdclasses = torch.tensor(self.fdist_classes, device=gt.device)
                scale_factor = gt.shape[-1] // feat[lay].shape[-1]
                gt_rescaled = downscale_label_ratio(gt, scale_factor,
                                                    self.fdist_scale_min_ratio,
                                                    self.num_classes,
                                                    255).long().detach()
                fdist_mask = torch.any(gt_rescaled[..., None] == fdclasses, -1)
                feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay],
                                                  fdist_mask)
                self.debug_fdist_mask = fdist_mask
                self.debug_gt_rescale = gt_rescaled
            else:
                feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay])
        feat_dist = self.fdist_lambda * feat_dist
        feat_loss, feat_log = self._parse_losses(
            {'loss_imnet_feat_dist': feat_dist})
        feat_log.pop('loss', None)
        return feat_loss, feat_log

    def update_debug_state(self):
        debug = self.local_iter % self.debug_img_interval == 0
        self.get_model().automatic_debug = False
        self.get_model().debug = debug
        if not self.source_only:
            self.get_ema_model().automatic_debug = False
            self.get_ema_model().debug = debug
        if self.mic is not None:
            self.mic.debug = debug

    def get_pseudo_label_and_weight(self, logits, sam_masks_batch):
        ema_softmax = torch.softmax(logits.detach(), dim=1)
        pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)
        pseudo_label = torch.where(pseudo_prob>0.5, pseudo_label, 13) # modified

        # SAM pseudo label refinement
        if self.is_SAM:
            if (self.max_iters/4)*(1-self.SAM_ratio) <=self.local_iter%(self.max_iters/4) <= (self.max_iters/4):
                new_pseudo_label = pseudo_label.clone() #torch.zeros_like(pseudo_label)
                for bb in range(len(sam_masks_batch)): # 2
                    for mask in sam_masks_batch[bb]: # 64:
                        mask_seg = torch.Tensor(mask['segmentation']).to(logits.device)
                        pixel_count = torch.count_nonzero(mask_seg)

                        pseudo_label_mask = torch.where(mask_seg==1, pseudo_label[bb], 255)
                        pixel_per_class = torch.bincount(pseudo_label_mask.flatten())
                        pixel_per_class = pixel_per_class[:14]# 0~13
                        max_class = torch.argmax(pixel_per_class)

                        new_pseudo_label[bb] = torch.where(mask_seg==1, max_class, new_pseudo_label[bb])
                pseudo_label = new_pseudo_label

        ps_large_p = pseudo_prob.ge(self.pseudo_threshold).long() == 1
        ps_size = np.size(np.array(pseudo_label.cpu()))
        pseudo_weight = torch.sum(ps_large_p).item() / ps_size
        pseudo_weight = pseudo_weight * torch.ones(
            pseudo_prob.shape, device=logits.device)
        return pseudo_label, pseudo_weight

    def filter_valid_pseudo_region(self, pseudo_weight, valid_pseudo_mask):
        if self.psweight_ignore_top > 0:
            # Don't trust pseudo-labels in regions with potential
            # rectification artifacts. This can lead to a pseudo-label
            # drift from sky towards building or traffic light.
            assert valid_pseudo_mask is None
            pseudo_weight[:, :self.psweight_ignore_top, :] = 0
        if self.psweight_ignore_bottom > 0:
            assert valid_pseudo_mask is None
            pseudo_weight[:, -self.psweight_ignore_bottom:, :] = 0
        if valid_pseudo_mask is not None:
            pseudo_weight *= valid_pseudo_mask.squeeze(1)
        return pseudo_weight

    def forward_train(self,
                      img,
                      img_metas,
                      gt_semantic_seg,
                      target_img,
                      target_img_metas,
                      rare_class=None,
                      valid_pseudo_mask=None):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        log_vars = {}
        batch_size = img.shape[0]
        dev = img.device

        # Init/update ema model
        if self.local_iter == 0:
            self._init_ema_weights()
            # assert _params_equal(self.get_ema_model(), self.get_model())

        if self.local_iter > 0:
            self._update_ema(self.local_iter)
            # assert not _params_equal(self.get_ema_model(), self.get_model())
            # assert self.get_ema_model().training
        if self.mic is not None:
            self.mic.update_weights(self.get_model(), self.local_iter)

        self.update_debug_state()
        seg_debug = {}

        means, stds = get_mean_std(img_metas, dev)
        strong_parameters = {
            'mix': None,
            'color_jitter': random.uniform(0, 1),
            'color_jitter_s': self.color_jitter_s,
            'color_jitter_p': self.color_jitter_p,
            'blur': random.uniform(0, 1) if self.blur else 0,
            'mean': means[0].unsqueeze(0),  # assume same normalization
            'std': stds[0].unsqueeze(0)
        }

        # modified
        pseudo_label, pseudo_weight = None, None
        if not self.source_only:
            # Generate pseudo-label
            for m in self.get_ema_model().modules():
                if isinstance(m, _DropoutNd):
                    m.training = False
                if isinstance(m, DropPath):
                    m.training = False
            ema_logits = self.get_ema_model().generate_pseudo_label(
                target_img, target_img_metas)
            seg_debug['Target'] = self.get_ema_model().debug_output


#######
            if self.is_SAM:
                if (self.max_iters/4)*(1-self.SAM_ratio) <=self.local_iter%(self.max_iters/4) <= (self.max_iters/4):
                    sam_trg_img = torch.clamp(denorm(target_img, means, stds), 0, 1) * 255
                    sam_masks_batch = []
                    for bb in range(target_img.shape[0]):
                        sam_trg_img_np = np.array(sam_trg_img[bb].permute(1,2,0).detach().cpu().numpy()).astype(np.uint8)
                        sam_masks_img = self.mask_generator.generate(sam_trg_img_np)
                        sam_masks_batch.append(sam_masks_img)
                else:
                    sam_masks_batch = None
            else:
                    sam_masks_batch = None


            # # SAM visualize
            # from PIL import Image
            # image_numpy = np.array(sam_masks[1]['segmentation'] * 255).astype(np.uint8)
            # image_numpy_to_PIL = Image.fromarray(image_numpy)
            # image_numpy_to_PIL.save('mask.png')
            # sam_trg_img.save('tgt.png')
##########

            pseudo_label, pseudo_weight = self.get_pseudo_label_and_weight(
                ema_logits, sam_masks_batch)
            del ema_logits

            pseudo_weight = self.filter_valid_pseudo_region(
                pseudo_weight, valid_pseudo_mask)
            gt_pixel_weight = torch.ones((pseudo_weight.shape), device=dev)

            # Apply mixing
            mixed_img, mixed_lbl = [None] * batch_size, [None] * batch_size
            mixed_seg_weight = pseudo_weight.clone()
            # mix_masks = get_class_masks(gt_semantic_seg)
####
            
            mix_masks, resize_class_= get_class_masks_resize(gt_semantic_seg)
            if self.is_shape:
                resize_gt_semantic_seg=resize(
                    input=gt_semantic_seg.float(),
                    size=None,
                    scale_factor=0.5,
                    #mode='bilinear',
                    mode='nearest',
                    align_corners=False)
                resize_img=resize(
                    input=img.float(),
                    size=None,
                    #mode='bilinear',
                    scale_factor=0.5,
                    mode='nearest',
                    align_corners=False)

                x_choice=np.random.choice(range(128,384),1,replace=False)[0]
                y_choice=np.random.choice(range(128,384),1,replace=False)[0]
                re_img=torch.zeros_like(img.long()) # 1,3,1024,1024
                re_img[:,:,x_choice-128:x_choice+128,y_choice-128:y_choice+128]=resize_img.long()
                re_gt=torch.zeros_like(gt_semantic_seg) # 1,1,1024,1024
                re_gt[:,:,x_choice-128:x_choice+128,y_choice-128:y_choice+128]=resize_gt_semantic_seg.long()

                resize_class_masks=[]
                for i in range(batch_size):
                    if type(resize_class_[i])!=int: # half classes
                        m=(re_gt[i]==1000).unsqueeze(1)
                        for j in resize_class_[i]:
                            mask=(re_gt[i]==j).unsqueeze(1)
                            m+=mask
                        masks=torch.where(m,1,0)
                        resize_class_masks.append(masks)
                    else:                           # 1 class
                        mask=(re_gt[i]==resize_class_[i]).unsqueeze(1)
                        masks=torch.where(mask,1,0)
                        resize_class_masks.append(masks)
#########


            for i in range(batch_size):
                strong_parameters['mix'] = mix_masks[i]
                mixed_img[i], mixed_lbl[i] = strong_transform(
                    strong_parameters,
                    data=torch.stack((img[i], target_img[i])),
                    target=torch.stack(
                        (gt_semantic_seg[i][0], pseudo_label[i])))
                _, mixed_seg_weight[i] = strong_transform(
                    strong_parameters,
                    target=torch.stack((gt_pixel_weight[i], pseudo_weight[i])))
            # del gt_pixel_weight
            mixed_img = torch.cat(mixed_img)
            mixed_lbl = torch.cat(mixed_lbl)

            if self.is_shape:
                re_mixed_img, re_mixed_lbl = [None] * batch_size, [None] * batch_size
                re_mixed_seg_weight = mixed_seg_weight.clone()
                for i in range(batch_size):
                    strong_parameters['mix'] = resize_class_masks[i]
                    re_mixed_img[i], re_mixed_lbl[i] = strong_transform(
                        strong_parameters,
                        data=torch.stack((re_img[i], mixed_img[i])),
                        target=torch.stack(
                            (re_gt[i][0], mixed_lbl[i][0])))
                    _, re_mixed_seg_weight[i] = strong_transform(
                        strong_parameters,
                        target=torch.stack((gt_pixel_weight[i], re_mixed_seg_weight[i])))
                #del gt_pixel_weight
                re_mixed_img = torch.cat(re_mixed_img)
                re_mixed_lbl = torch.cat(re_mixed_lbl)
            else:
                re_mixed_img=mixed_img
                re_mixed_lbl=mixed_lbl
                re_mixed_seg_weight=mixed_seg_weight



############### dilation loss
            if self.is_dilation:
                kernel = np.ones((3,3), np.uint8)
                private_mask_ori = torch.where(re_mixed_lbl==13, 1, 0) # [2,1,1024,1024]

                private_masks, erosion_masks, dilation_masks = [None]*batch_size, [None]*batch_size, [None]*batch_size
                for bb in range(batch_size):
                    private_mask = private_mask_ori.detach().cpu().numpy()[bb].transpose(1,2,0)
                    private_mask = np.array(private_mask * 255, dtype = np.uint8)
                    # cv2.imwrite('private.jpg', private_mask)

                    erosion_mask = cv2.erode(private_mask, kernel=kernel, iterations=1)
                    # cv2.imwrite('erosion.jpg', erosion_mask)
                    erosion_mask = erosion_mask.astype(np.uint8)

                    dilation_mask = cv2.dilate(private_mask, kernel=kernel, iterations=1)
                    # cv2.imwrite('dilation.jpg', dilation_mask)
                    dilation_mask = dilation_mask.astype(np.uint8)

                    private_mask = torch.from_numpy(private_mask).unsqueeze(0)
                    erosion_mask = torch.from_numpy(erosion_mask).unsqueeze(0)
                    dilation_mask = torch.from_numpy(dilation_mask).unsqueeze(0)
                    private_masks[bb] = private_mask
                    erosion_masks[bb] = erosion_mask
                    dilation_masks[bb] = dilation_mask

                private_masks = torch.cat(private_masks).permute(0,3,1,2)
                erosion_masks = torch.cat(erosion_masks).unsqueeze(1)
                dilation_masks = torch.cat(dilation_masks).unsqueeze(1)
                dilation_mask = [private_masks, erosion_masks, dilation_masks]
            else: 
                dilation_mask=None


            # # Train on mixed images
            # mix_losses = self.get_model().forward_train(
            #     mixed_img,
            #     img_metas,
            #     mixed_lbl,
            #     dilation_mask = dilation_mask,
            #     seg_weight=mixed_seg_weight,
            #     return_feat=False,
            # )
            # seg_debug['Mix'] = self.get_model().debug_output
            # mix_losses = add_prefix(mix_losses, 'mix')
            # mix_loss, mix_log_vars = self._parse_losses(mix_losses)
            # log_vars.update(mix_log_vars)
            # mix_loss.backward()

            # Train on mixed images
            mix_losses = self.get_model().forward_train(
                re_mixed_img,
                img_metas,
                re_mixed_lbl,
                dilation_mask = dilation_mask,
                seg_weight=re_mixed_seg_weight,
                return_feat=False,
            )
            seg_debug['Mix'] = self.get_model().debug_output
            mix_losses = add_prefix(mix_losses, 'mix')
            mix_loss, mix_log_vars = self._parse_losses(mix_losses)
            log_vars.update(mix_log_vars)
            mix_loss.backward()
        del mix_loss,dilation_mask,re_mixed_seg_weight
        # Apply source private mixing
        #if 7000<=self.local_iter<=10000 or 17000<=self.local_iter<=20000 or 27000<=self.local_iter<=30000 or 37000<=self.local_iter<=40000:
        if self.is_bimix:
            if (self.max_iters/4)*(1-self.SAM_ratio) <=self.local_iter%(self.max_iters/4) <= (self.max_iters/4):
                source_mixed_img, source_mixed_lbl = [None] * batch_size, [None] * batch_size
                source_mixed_seg_weight = pseudo_weight.clone()
                source_mix_masks = (pseudo_label==13)
                source_mix_weight=torch.where(source_mix_masks,pseudo_weight,gt_pixel_weight)
                source_mix_masks=source_mix_masks.unsqueeze(1)
                m_pseudo_label=pseudo_label.unsqueeze(1)
                new_gt_semantic_seg=torch.where(source_mix_masks,m_pseudo_label,gt_semantic_seg)
                new_source_img=torch.where(source_mix_masks,target_img,img)
                
                gt_pixel_weight=source_mixed_seg_weight
                img=new_source_img
                gt_semantic_seg=new_gt_semantic_seg
                del new_source_img,source_mixed_seg_weight,new_gt_semantic_seg


        # Train on source images
        clean_losses = self.get_model().forward_train(
            img, img_metas, gt_semantic_seg,seg_weight=gt_pixel_weight,return_feat=True)
        src_feat = clean_losses.pop('features')
        seg_debug['Source'] = self.get_model().debug_output
        clean_loss, clean_log_vars = self._parse_losses(clean_losses)
        log_vars.update(clean_log_vars)
        clean_loss.backward(retain_graph=self.enable_fdist)
        if self.print_grad_magnitude:
            params = self.get_model().backbone.parameters()
            seg_grads = [
                p.grad.detach().clone() for p in params if p.grad is not None
            ]
            grad_mag = calc_grad_magnitude(seg_grads)
            mmcv.print_log(f'Seg. Grad.: {grad_mag}', 'mmseg')

        # ImageNet feature distance
        if self.enable_fdist:
            feat_loss, feat_log = self.calc_feat_dist(img, gt_semantic_seg,
                                                      src_feat)
            log_vars.update(add_prefix(feat_log, 'src'))
            feat_loss.backward()
            if self.print_grad_magnitude:
                params = self.get_model().backbone.parameters()
                fd_grads = [
                    p.grad.detach() for p in params if p.grad is not None
                ]
                fd_grads = [g2 - g1 for g1, g2 in zip(seg_grads, fd_grads)]
                grad_mag = calc_grad_magnitude(fd_grads)
                mmcv.print_log(f'Fdist Grad.: {grad_mag}', 'mmseg')
        del src_feat, clean_loss
        if self.enable_fdist:
            del feat_loss

        del gt_pixel_weight ,re_mixed_img,re_mixed_lbl

        # Masked Training
        if self.enable_masking and self.mask_mode.startswith('separate'):
            masked_loss = self.mic(self.get_model(), img, img_metas,
                                   gt_semantic_seg, target_img,
                                   target_img_metas, valid_pseudo_mask,
                                   pseudo_label, pseudo_weight)
            seg_debug.update(self.mic.debug_output)
            masked_loss = add_prefix(masked_loss, 'masked')
            masked_loss, masked_log_vars = self._parse_losses(masked_loss)
            log_vars.update(masked_log_vars)
            masked_loss.backward()
        
        if self.local_iter % self.debug_img_interval == 0 and \
                not self.source_only:
            out_dir = os.path.join(self.train_cfg['work_dir'], 'debug')
            os.makedirs(out_dir, exist_ok=True)
            vis_img = torch.clamp(denorm(img, means, stds), 0, 1)
            vis_trg_img = torch.clamp(denorm(target_img, means, stds), 0, 1)
            vis_mixed_img = torch.clamp(denorm(mixed_img, means, stds), 0, 1)
            for j in range(batch_size):
                rows, cols = 2, 5
                fig, axs = plt.subplots(
                    rows,
                    cols,
                    figsize=(3 * cols, 3 * rows),
                    gridspec_kw={
                        'hspace': 0.1,
                        'wspace': 0,
                        'top': 0.95,
                        'bottom': 0,
                        'right': 1,
                        'left': 0
                    },
                )
                subplotimg(axs[0][0], vis_img[j], 'Source Image')
                subplotimg(axs[1][0], vis_trg_img[j], 'Target Image')
                subplotimg(
                    axs[0][1],
                    gt_semantic_seg[j],
                    'Source Seg GT',
                    cmap='cityscapes')
                subplotimg(
                    axs[1][1],
                    pseudo_label[j],
                    'Target Seg (Pseudo) GT',
                    cmap='cityscapes')
                subplotimg(axs[0][2], vis_mixed_img[j], 'Mixed Image')
                subplotimg(
                    axs[1][2], mix_masks[j][0], 'Domain Mask', cmap='gray')
                # subplotimg(axs[0][3], pred_u_s[j], "Seg Pred",
                #            cmap="cityscapes")
                if mixed_lbl is not None:
                    subplotimg(
                        axs[1][3], mixed_lbl[j], 'Seg Targ', cmap='cityscapes')
                subplotimg(
                    axs[0][3],
                    mixed_seg_weight[j],
                    'Pseudo W.',
                    vmin=0,
                    vmax=1)
                if self.debug_fdist_mask is not None:
                    subplotimg(
                        axs[0][4],
                        self.debug_fdist_mask[j][0],
                        'FDist Mask',
                        cmap='gray')
                if self.debug_gt_rescale is not None:
                    subplotimg(
                        axs[1][4],
                        self.debug_gt_rescale[j],
                        'Scaled GT',
                        cmap='cityscapes')
                for ax in axs.flat:
                    ax.axis('off')
                plt.savefig(
                    os.path.join(out_dir,
                                 f'{(self.local_iter + 1):06d}_{j}.png'))
                plt.close()

        if self.local_iter % self.debug_img_interval == 0:
            out_dir = os.path.join(self.train_cfg['work_dir'], 'debug')
            os.makedirs(out_dir, exist_ok=True)
            if seg_debug['Source'] is not None and seg_debug:
                if 'Target' in seg_debug:
                    seg_debug['Target']['Pseudo W.'] = mixed_seg_weight.cpu(
                    ).numpy()
                for j in range(batch_size):
                    cols = len(seg_debug)
                    rows = max(len(seg_debug[k]) for k in seg_debug.keys())
                    fig, axs = plt.subplots(
                        rows,
                        cols,
                        figsize=(5 * cols, 5 * rows),
                        gridspec_kw={
                            'hspace': 0.1,
                            'wspace': 0,
                            'top': 0.95,
                            'bottom': 0,
                            'right': 1,
                            'left': 0
                        },
                        squeeze=False,
                    )
                    for k1, (n1, outs) in enumerate(seg_debug.items()):
                        for k2, (n2, out) in enumerate(outs.items()):
                            subplotimg(
                                axs[k2][k1],
                                **prepare_debug_out(f'{n1} {n2}', out[j],
                                                    means, stds))
                    for ax in axs.flat:
                        ax.axis('off')
                    plt.savefig(
                        os.path.join(out_dir,
                                     f'{(self.local_iter + 1):06d}_{j}_s.png'))
                    plt.close()
                del seg_debug
        self.local_iter += 1

        return log_vars
