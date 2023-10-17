# Obtained from: https://github.com/lhoyer/HRDA
# Modifications:
# - Add return_logits flag
# - Update debug_output
# ---------------------------------------------------------------
# Copyright (c) 2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

from copy import deepcopy

import torch
from torch.nn import functional as F

from ...core import add_prefix
from ...ops import resize as _resize
from .. import builder
from ..builder import HEADS
from ..segmentors.hrda_encoder_decoder import crop
from .decode_head import BaseDecodeHead


def scale_box(box, scale):
    y1, y2, x1, x2 = box
    # assert y1 % scale == 0
    # assert y2 % scale == 0
    # assert x1 % scale == 0
    # assert x2 % scale == 0
    y1 = int(y1 / scale)
    y2 = int(y2 / scale)
    x1 = int(x1 / scale)
    x2 = int(x2 / scale)
    return y1, y2, x1, x2


@HEADS.register_module()
class HRDAHead(BaseDecodeHead):

    def __init__(self,
                 single_scale_head,
                 lr_loss_weight=0,
                 hr_loss_weight=0,
                 scales=[1],
                 attention_embed_dim=256,
                 attention_classwise=True,
                 enable_hr_crop=False,
                 hr_slide_inference=True,
                 fixed_attention=None,
                 debug_output_attention=False,
                 **kwargs):
        head_cfg = deepcopy(kwargs)
        attn_cfg = deepcopy(kwargs)
        if single_scale_head == 'DAFormerHead':
            attn_cfg['channels'] = attention_embed_dim
            attn_cfg['decoder_params']['embed_dims'] = attention_embed_dim
            if attn_cfg['decoder_params']['fusion_cfg']['type'] == 'aspp':
                attn_cfg['decoder_params']['fusion_cfg'] = dict(
                    type='conv',
                    kernel_size=1,
                    act_cfg=dict(type='ReLU'),
                    norm_cfg=attn_cfg['decoder_params']['fusion_cfg']
                    ['norm_cfg'])
            kwargs['init_cfg'] = None
            kwargs['input_transform'] = 'multiple_select'
            self.os = 4
        elif single_scale_head == 'DLV2Head':
            kwargs['init_cfg'] = None
            kwargs.pop('dilations')
            kwargs['channels'] = 1
            self.os = 8
        else:
            raise NotImplementedError(single_scale_head)
        super(HRDAHead, self).__init__(**kwargs)
        del self.conv_seg
        del self.dropout

        head_cfg['type'] = single_scale_head
        self.head = builder.build_head(head_cfg)

        attn_cfg['type'] = single_scale_head
        if not attention_classwise:
            attn_cfg['num_classes'] = 1
        if fixed_attention is None:
            self.scale_attention = builder.build_head(attn_cfg)
        else:
            self.scale_attention = None
            self.fixed_attention = fixed_attention
        self.lr_loss_weight = lr_loss_weight
        self.hr_loss_weight = hr_loss_weight
        self.scales = scales
        self.enable_hr_crop = enable_hr_crop
        self.hr_crop_box = None
        self.hr_slide_inference = hr_slide_inference
        self.debug_output_attention = debug_output_attention

        self.ce_criterion = torch.nn.CrossEntropyLoss() # modified


    def set_hr_crop_box(self, boxes):
        self.hr_crop_box = boxes

    def hr_crop_slice(self, scale):
        crop_y1, crop_y2, crop_x1, crop_x2 = scale_box(self.hr_crop_box, scale)
        return slice(crop_y1, crop_y2), slice(crop_x1, crop_x2)

    def resize(self, input, scale_factor):
        return _resize(
            input=input,
            scale_factor=scale_factor,
            mode='nearest')

    def decode_hr(self, inp, bs):
        if isinstance(inp, dict) and 'boxes' in inp.keys():
            features = inp['features']  # level, crop * bs, c, h, w
            boxes = inp['boxes']
            dev = features[0][0].device
            h_img, w_img = 0, 0
            for i in range(len(boxes)):
                boxes[i] = scale_box(boxes[i], self.os)
                y1, y2, x1, x2 = boxes[i]
                if h_img < y2:
                    h_img = y2
                if w_img < x2:
                    w_img = x2
            preds = torch.zeros((bs, self.num_classes, h_img, w_img),
                                device=dev)
            preds_feat = torch.zeros((bs, 256, h_img, w_img),
                                device=dev)
            count_mat = torch.zeros((bs, 1, h_img, w_img), device=dev)

            crop_seg_feats, crop_seg_logits = self.head(features) # modified
            for i in range(len(boxes)):
                y1, y2, x1, x2 = boxes[i]
                crop_seg_logit = crop_seg_logits[i * bs:(i + 1) * bs]
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                crop_seg_feat = crop_seg_feats[i * bs:(i + 1) * bs]
                preds_feat += F.pad(crop_seg_feat,
                               (int(x1), int(preds_feat.shape[3] - x2), int(y1),
                                int(preds_feat.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1

            assert (count_mat == 0).sum() == 0
            preds = preds / count_mat
            preds_feat = preds_feat / count_mat
            return preds_feat, preds 
        else:
            return self.head(inp)

    def get_scale_attention(self, inp):
        if self.scale_attention is not None:
            # att = torch.sigmoid(self.scale_attention(inp))
            att = torch.sigmoid(self.scale_attention(inp)[1])
        else:
            att = self.fixed_attention
        return att

    def forward(self, inputs, dilation_mask):
        assert len(inputs) == 2
        hr_inp = inputs[1]
        hr_scale = self.scales[1]
        lr_inp = inputs[0]
        lr_sc_att_inp = inputs[0]  # separate var necessary for stack hr_fusion
        lr_scale = self.scales[0]
        batch_size = lr_inp[0].shape[0]
        assert lr_scale <= hr_scale

        has_crop = self.hr_crop_box is not None
        if has_crop:
            crop_y1, crop_y2, crop_x1, crop_x2 = self.hr_crop_box

        # print_log(f'lr_inp {[f.shape for f in lr_inp]}', 'mmseg')
        lr_feat, lr_seg = self.head(lr_inp)
        # print_log(f'lr_seg {lr_seg.shape}', 'mmseg')

        hr_feat, hr_seg = self.decode_hr(hr_inp, batch_size)

        att = self.get_scale_attention(lr_sc_att_inp)
        att_feat = att[:,0,:,:].unsqueeze(1).repeat(1,lr_feat.shape[1],1,1)
        if has_crop:
            mask = lr_seg.new_zeros([lr_seg.shape[0], 1, *lr_seg.shape[2:]])
            sc_os = self.os / lr_scale
            slc = self.hr_crop_slice(sc_os)
            mask[:, :, slc[0], slc[1]] = 1
            att = att * mask

            mask_feat = lr_feat.new_zeros([lr_feat.shape[0], 1, *lr_feat.shape[2:]])
            sc_os = self.os / lr_scale
            slc = self.hr_crop_slice(sc_os)
            mask_feat[:, :, slc[0], slc[1]] = 1
            att_feat = att_feat * mask_feat        # print_log(f'att {att.shape}', 'mmseg')

        lr_seg = (1 - att) * lr_seg
        lr_feat = (1 - att_feat) * lr_feat
        # print_log(f'scaled lr_seg {lr_seg.shape}', 'mmseg')
        up_lr_seg = self.resize(lr_seg, hr_scale / lr_scale)
        up_lr_feat = self.resize(lr_feat, hr_scale / lr_scale) # modified
        if torch.is_tensor(att):
            att = self.resize(att, hr_scale / lr_scale)
            att_feat = self.resize(att_feat, hr_scale / lr_scale)

        if has_crop:
            hr_seg_inserted = torch.zeros_like(up_lr_seg)
            slc = self.hr_crop_slice(self.os)
            hr_seg_inserted[:, :, slc[0], slc[1]] = hr_seg


            hr_feat_inserted = torch.zeros_like(up_lr_feat)
            slc = self.hr_crop_slice(self.os)
            hr_feat_inserted[:, :, slc[0], slc[1]] = hr_feat
        else:
            hr_seg_inserted = hr_seg
            hr_feat_inserted = hr_feat

        fused_seg = att * hr_seg_inserted + up_lr_seg
        fused_feat = att_feat * hr_feat_inserted + up_lr_feat
        
        if self.debug_output_attention:
            att = torch.sum(
                att * torch.softmax(fused_seg, dim=1), dim=1, keepdim=True)
            return att, None, None

        if self.debug:
            self.debug_output.update({
                'High Res':
                torch.max(hr_seg, dim=1)[1].detach().cpu().numpy(),
                'High Res Inserted':
                torch.max(hr_seg_inserted, dim=1)[1].detach().cpu().numpy(),
                'Low Res':
                torch.max(lr_seg, dim=1)[1].detach().cpu().numpy(),
                'Fused':
                torch.max(fused_seg, dim=1)[1].detach().cpu().numpy(),
            })
            if torch.is_tensor(att):
                self.debug_output['Attention'] = torch.sum(
                    att * torch.softmax(fused_seg, dim=1), dim=1,
                    keepdim=True).detach().cpu().numpy()

        return fused_seg, lr_seg, hr_seg, fused_feat

    def reset_crop(self):
        del self.hr_crop_box
        self.hr_crop_box = None

    def forward_train(self,
                      inputs,
                      img_metas,
                      gt_semantic_seg,
                      dilation_mask,
                      train_cfg,
                      seg_weight=None,
                      return_logits=False):
        """Forward function for training."""
        if self.enable_hr_crop:
            assert self.hr_crop_box is not None
        seg_logits = self.forward(inputs, dilation_mask)
        losses = self.losses(seg_logits, gt_semantic_seg, dilation_mask, seg_weight)
        if return_logits:
            losses['logits'] = seg_logits
        self.reset_crop()
        return losses

    def forward_test(self, inputs, img_metas, dilation_mask, test_cfg):
        """Forward function for testing, only ``fused_seg`` is used."""
        return self.forward(inputs, dilation_mask)[0]

    def losses(self, seg_logit, seg_label, dilation_mask, seg_weight=None):
        """Compute losses."""
        fused_seg, lr_seg, hr_seg, fused_feat = seg_logit # modified
        loss = super(HRDAHead, self).losses(fused_seg, seg_label, seg_weight)
        if self.hr_loss_weight == 0 and self.lr_loss_weight == 0:
            return loss

        if self.lr_loss_weight > 0:
            loss.update(
                add_prefix(
                    super(HRDAHead, self).losses(lr_seg, seg_label,
                                                 seg_weight), 'lr'))
        if self.hr_loss_weight > 0 and self.enable_hr_crop:
            cropped_seg_label = crop(seg_label, self.hr_crop_box)
            if seg_weight is not None:
                cropped_seg_weight = crop(seg_weight, self.hr_crop_box)
            else:
                cropped_seg_weight = seg_weight
            if self.debug:
                self.debug_output['Cropped GT'] = \
                    cropped_seg_label.squeeze(1).detach().cpu().numpy()
            loss.update(
                add_prefix(
                    super(HRDAHead, self).losses(hr_seg, cropped_seg_label,
                                                 cropped_seg_weight), 'hr'))
        elif self.hr_loss_weight > 0:
            loss.update(
                add_prefix(
                    super(HRDAHead, self).losses(hr_seg, seg_label,
                                                 seg_weight), 'hr'))
#######

        if dilation_mask is not None:
            D = fused_feat.shape[1]
            # fused_feat [2,256,256,256]
            private_mask, erosion_mask, dilation_mask = dilation_mask
            private_mask = self.resize(private_mask, 0.25)
            erosion_mask = self.resize(erosion_mask, 0.25)
            dilation_mask = self.resize(dilation_mask, 0.25) # [B,1,256,256]

            # import cv2
            # import numpy as np
            # private_mask = private_mask.detach().cpu().numpy()
            # private_mask = np.array(private_mask, dtype = np.uint8)

            # erosion_mask = erosion_mask.detach().cpu().numpy()
            # erosion_mask = np.array(erosion_mask, dtype = np.uint8)

            # dilation_mask = dilation_mask.detach().cpu().numpy()
            # dilation_mask = np.array(dilation_mask, dtype = np.uint8)

            # cv2.imwrite('private2.jpg', private_mask[0].transpose(1,2,0))
            # cv2.imwrite('erosion2.jpg', erosion_mask[0].transpose(1,2,0))
            # cv2.imwrite('dilation2.jpg', dilation_mask[0].transpose(1,2,0))
            # print("Visualization Done")

            private_mask = torch.where(private_mask==255, 1, 0) # 255->1
            erosion_mask = torch.where(erosion_mask==255, 1, 0)
            dilation_mask = torch.where(dilation_mask==255, 1, 0)
            common_mask = torch.logical_and((1-private_mask), dilation_mask) # [B,1,256,256]

            # private_mask = private_mask.reshape(private_mask.shape[1], -1).repeat(fused_feat.shape[1], 1)
            # erosion_mask = erosion_mask.reshape(erosion_mask.shape[1], -1).repeat(fused_feat.shape[1], 1)
            # dilation_mask = dilation_mask.reshape(dilation_mask.shape[1], -1).repeat(fused_feat.shape[1], 1)
            # fused_feat = fused_feat.reshape(fused_feat.shape[1], -1)

            # private_mask = private_mask[:,:1000]
            # erosion_mask = erosion_mask[:,:1000]
            # dilation_mask = dilation_mask[:,:1000]
            # common_mask = common_mask[:,:1000]
            # fused_feat = fused_feat[:,:1000] # [256,1000]

            """
            h, w = torch.randint(64,192,(1,)), torch.randint(64,192,(1,))
            p=64      
            private_mask = private_mask[:,:,h-p:h+p,w-p:w+p].reshape(-1, 1).repeat(1,D)
            erosion_mask = erosion_mask[:,:,h-p:h+p,w-p:w+p].reshape(-1, 1).repeat(1,D)
            dilation_mask = dilation_mask[:,:,h-p:h+p,w-p:w+p].reshape(-1, 1).repeat(1,D)
            common_mask = common_mask[:,:,h-p:h+p,w-p:w+p].reshape(-1, 1).repeat(1,D) # [*,1] -> [*,256]
            fused_feat = fused_feat[:,:,h-p:h+p,w-p:w+p].reshape(-1, D) # [*,256]

            if torch.count_nonzero(private_mask)>0 and torch.count_nonzero(erosion_mask)>0 and torch.count_nonzero(dilation_mask)>0 and torch.count_nonzero(common_mask)>0:
                private_feat = fused_feat[private_mask!=0].reshape(-1, D)

                private_proto = fused_feat[erosion_mask!=0].reshape(-1, D)
                private_proto = torch.mean(private_proto, dim=0, keepdim=True)
                common_feat = fused_feat[common_mask!=0].reshape(-1, D)
                target_feat = torch.cat([private_proto, common_feat], dim=0) 

                logits = torch.matmul(F.normalize(private_feat, dim=-1), F.normalize(target_feat, dim=-1).T)
                labels = torch.zeros(logits.shape[0], dtype=torch.long).to(fused_feat.device)

                loss_dilation = self.ce_criterion(logits, labels)
                loss.update(add_prefix({'loss_dilation':loss_dilation}, 'dilation'))
            """
            
            loss_dilation=0
            cnt=0
            #h, w = torch.randint(64,192,(1,)), torch.randint(64,192,(1,))
            #p=64
            h, w = torch.randint(32,96,(1,)), torch.randint(32,96,(1,))
            p=32      
            private_mask2 = private_mask[:,:,h-p:h+p,w-p:w+p].reshape(-1, 1).repeat(1,D)
            erosion_mask2 = erosion_mask[:,:,h-p:h+p,w-p:w+p].reshape(-1, 1).repeat(1,D)
            dilation_mask2 = dilation_mask[:,:,h-p:h+p,w-p:w+p].reshape(-1, 1).repeat(1,D)
            common_mask2 = common_mask[:,:,h-p:h+p,w-p:w+p].reshape(-1, 1).repeat(1,D) # [*,1] -> [*,256]
            fused_feat2 = fused_feat[:,:,h-p:h+p,w-p:w+p].reshape(-1, D) # [*,256]

            if torch.count_nonzero(private_mask2)>0 and torch.count_nonzero(erosion_mask2)>0 and torch.count_nonzero(dilation_mask2)>0 and torch.count_nonzero(common_mask2)>0:
                cnt+=1
                private_feat = fused_feat2[private_mask2!=0].reshape(-1, D)

                private_proto = fused_feat2[erosion_mask2!=0].reshape(-1, D)
                private_proto = torch.mean(private_proto, dim=0, keepdim=True)
                common_feat = fused_feat2[common_mask2!=0].reshape(-1, D)
                target_feat = torch.cat([private_proto, common_feat], dim=0) 

                logits = torch.matmul(F.normalize(private_feat, dim=-1), F.normalize(target_feat, dim=-1).T)
                labels = torch.zeros(logits.shape[0], dtype=torch.long).to(fused_feat2.device)

                loss_dilation = self.ce_criterion(logits, labels) / common_mask.size()[0]
                loss.update(add_prefix({'loss_dilation':loss_dilation}, 'dilation'))
            


        loss['loss_seg'] *= (1 - self.lr_loss_weight - self.hr_loss_weight)
        if self.lr_loss_weight > 0:
            loss['lr.loss_seg'] *= self.lr_loss_weight
        if self.hr_loss_weight > 0:
            loss['hr.loss_seg'] *= self.hr_loss_weight

        if self.debug:
            self.debug_output['GT'] = \
                seg_label.squeeze(1).detach().cpu().numpy()
            # Remove debug output from cross entropy loss
            self.debug_output.pop('Seg. Pred.', None)
            self.debug_output.pop('Seg. GT', None)

        return loss
