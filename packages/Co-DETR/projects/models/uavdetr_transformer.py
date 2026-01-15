
import torch
import torch.nn as nn
import torch.nn.functional as F

from .rtdetr_layer import ConvNormLayer, CSPRepLayer
from .uav_detr_block import SemanticAlignmenCalibration, FrequencyFocusedDownSampling, DySample, MFFF, Focus
from .co_deformable_rtdetr_transformer import CoRTDetrDeformableTransformer

from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
from mmcv.cnn import build_norm_layer

from fairscale.nn.checkpoint import checkpoint_wrapper

from mmdet.models.utils.transformer import Transformer, DeformableDetrTransformer, DeformableDetrTransformerDecoder
from mmdet.models.utils.builder import TRANSFORMER

def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.

    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)

@TRANSFORMER.register_module()
class CoDinoUAVDetrTransformer(CoRTDetrDeformableTransformer):

    def __init__(self, *args, **kwargs):
        super(CoDinoUAVDetrTransformer, self).__init__(*args, **kwargs)

    def init_layers(self):
        """Initialize layers of the DinoTransformer."""
        self.level_embeds = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.enc_output = nn.Linear(self.embed_dims, self.embed_dims)
        self.enc_output_norm = nn.LayerNorm(self.embed_dims)
        self.query_embed = nn.Embedding(self.two_stage_num_proposals,
                                        self.embed_dims)
    
    def _init_layers(self):
        if self.with_pos_coord:
            if self.num_co_heads > 0:
                self.aux_pos_trans = nn.ModuleList()
                self.aux_pos_trans_norm = nn.ModuleList()
                self.pos_feats_trans = nn.ModuleList()
                self.pos_feats_norm = nn.ModuleList()
                for i in range(self.num_co_heads):
                    self.aux_pos_trans.append(nn.Linear(self.embed_dims*2, self.embed_dims))
                    self.aux_pos_trans_norm.append(nn.LayerNorm(self.embed_dims))
                    if self.with_coord_feat:
                        self.pos_feats_trans.append(nn.Linear(self.embed_dims, self.embed_dims))
                        self.pos_feats_norm.append(nn.LayerNorm(self.embed_dims))

    def init_weights(self):
        super().init_weights()
        nn.init.normal_(self.query_embed.weight.data)

    def forward(self,
                mlvl_feats,
                mlvl_masks,
                query_embed,
                mlvl_pos_embeds,
                dn_label_query,
                dn_bbox_query,
                attn_mask,
                reg_branches=None,
                cls_branches=None,
                **kwargs):
        assert self.as_two_stage and query_embed is None, \
            'as_two_stage must be True for DINO'

        ccff_feats = mlvl_feats
        mask_flatten_enc_out = []

        for lvl, (mask) in enumerate(mlvl_masks[1:]):
            mask = mask.flatten(1)
            mask_flatten_enc_out.append(mask)
        
        mask_flatten_enc_out = torch.cat(mask_flatten_enc_out, 1)

        #sprint("mask flatten out", mask_flatten_enc_out.shape)

        mlvl_feats, mlvl_masks, mlvl_pos_embeds = [mlvl_feats[-1]], [mlvl_masks[-1]], [mlvl_pos_embeds[-1]]

        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
                zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            feat = feat.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack(
            [self.get_valid_ratio(m) for m in mlvl_masks], 1)

        reference_points = self.get_reference_points(
            spatial_shapes, valid_ratios, device=feat.device)

        feat_flatten = feat_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims)
        lvl_pos_embed_flatten = lvl_pos_embed_flatten.permute(
            1, 0, 2)  # (H*W, bs, embed_dims)
        memory, spatial_shapes, level_start_index = self.encoder(
            query=feat_flatten,
            key=None,
            value=None,
            ccff_feats=ccff_feats,
            query_pos=lvl_pos_embed_flatten,
            query_key_padding_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            **kwargs)
        
        #print("enc memory", memory.shape)
        #print("enc spatial shape", spatial_shapes)

        #memory = memory.permute(1, 0, 2)
        bs, _, c = memory.shape

        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, mask_flatten_enc_out, spatial_shapes)
        enc_outputs_class = cls_branches[self.decoder.num_layers](
            output_memory)
        enc_outputs_coord_unact = reg_branches[self.decoder.num_layers](
            output_memory) + output_proposals
        cls_out_features = cls_branches[self.decoder.num_layers].out_features
        topk = self.two_stage_num_proposals
        # NOTE In DeformDETR, enc_outputs_class[..., 0] is used for topk TODO
        topk_indices = torch.topk(enc_outputs_class.max(-1)[0], topk, dim=1)[1]

        topk_score = torch.gather(
            enc_outputs_class, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 4))
        topk_anchor = topk_coords_unact.sigmoid()
        topk_coords_unact = topk_coords_unact.detach()

        query = self.query_embed.weight[:, None, :].repeat(1, bs,
                                                           1).transpose(0, 1)
        # NOTE the query_embed here is not spatial query as in DETR.
        # It is actually content query, which is named tgt in other
        # DETR-like models
        if dn_label_query is not None:
            query = torch.cat([dn_label_query, query], dim=1)
        if dn_bbox_query is not None:
            reference_points = torch.cat([dn_bbox_query, topk_coords_unact],
                                         dim=1)
        else:
            reference_points = topk_coords_unact
        reference_points = reference_points.sigmoid()

        # decoder
        query = query.permute(1, 0, 2)
        memory = memory.permute(1, 0, 2)
        spatial_shapes = torch.tensor(spatial_shapes, dtype=torch.long, device=memory.device)
        level_start_index = torch.tensor(level_start_index, device=memory.device)

        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=memory,
            attn_masks=attn_mask,
            key_padding_mask=mask_flatten_enc_out,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=reg_branches,
            **kwargs)

        inter_references_out = inter_references

        return inter_states, inter_references_out, topk_score, topk_anchor, memory


    def forward_aux(self,
                    mlvl_feats,
                    mlvl_masks,
                    query_embed,
                    mlvl_pos_embeds,
                    pos_anchors,
                    pos_feats=None,
                    reg_branches=None,
                    cls_branches=None,
                    return_encoder_output=False,
                    attn_masks=None,
                    head_idx=0,
                    **kwargs):
        feat_flatten = []
        mask_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
                zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            feat = feat.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack(
            [self.get_valid_ratio(m) for m in mlvl_masks], 1)

        feat_flatten = feat_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims)
        
        memory = feat_flatten
            #enc_inter = [feat.permute(1, 2, 0) for feat in enc_inter]
        memory = memory.permute(1, 0, 2)
        bs, _, c = memory.shape

        topk = pos_anchors.shape[1]
        topk_coords_unact = inverse_sigmoid((pos_anchors))
        reference_points = (pos_anchors)
        init_reference_out = reference_points
        if self.num_co_heads > 0:
            pos_trans_out = self.aux_pos_trans_norm[head_idx](
                self.aux_pos_trans[head_idx](self.get_proposal_pos_embed(topk_coords_unact)))
            query = pos_trans_out
            if self.with_coord_feat:
                query = query + self.pos_feats_norm[head_idx](self.pos_feats_trans[head_idx](pos_feats))

        # decoder
        query = query.permute(1, 0, 2)
        memory = memory.permute(1, 0, 2)
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=memory,
            attn_masks=None,
            key_padding_mask=mask_flatten,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=reg_branches,
            **kwargs)

        inter_references_out = inter_references

        return inter_states, inter_references_out

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class UAVdetrTransformerEncoder(TransformerLayerSequence):
    """TransformerEncoder of DETR.

    Args:
        post_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`. Only used when `self.pre_norm` is `True`
    """

    def __init__(self, 
                 *args, 
                 post_norm_cfg=dict(type='LN'), 
                 with_cp=-1,
                 in_channels_ccff=[512, 1024, 2048], 
                 hidden_dim=256,
                 depth_mult=1,
                 act='silu',
                 expansion=1,
                 eval_spatial_size=[640, 640],
                 feat_strides=[8, 16, 32, 64],
                 pe_temperature=10000,
                 use_encoder_idx=[2],
                 **kwargs):
        super(UAVdetrTransformerEncoder, self).__init__(*args, **kwargs)
        
        # parameter tambahan Co-UAV Detr / CCFF
        self.hidden_dim=hidden_dim
        self.in_channels_ccff=in_channels_ccff
        self.use_encoder_idx = use_encoder_idx
        self.eval_spatial_size = eval_spatial_size
        self.feat_strides = feat_strides
        self.pe_temperature = pe_temperature


        if post_norm_cfg is not None:
            self.post_norm = build_norm_layer(
                post_norm_cfg, self.embed_dims)[1] if self.pre_norm else None
        else:
            assert not self.pre_norm, f'Use prenorm in ' \
                                      f'{self.__class__.__name__},' \
                                      f'Please specify post_norm_cfg'
            self.post_norm = None
        self.with_cp = with_cp
        if self.with_cp > 0:
            for i in range(self.with_cp):
                self.layers[i] = checkpoint_wrapper(self.layers[i])

        # top-down fpn
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        self.upsample_conv = nn.ModuleList()
        for _ in range(len(in_channels_ccff) - 1, 2, -1):
            self.lateral_convs.append(ConvNormLayer(hidden_dim, hidden_dim, 1, 1, act=act))
            self.fpn_blocks.append(
                CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
            )
            self.upsample_conv.append(DySample(hidden_dim, scale=2, style="lp"))

        # bottom-up pan
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(in_channels_ccff) - 2):
            self.downsample_convs.append(
                FrequencyFocusedDownSampling(hidden_dim, hidden_dim)
            )
            self.pan_blocks.append(
                CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
            )
        
        self.mfff = MFFF(hidden_dim*3)
        self.lateral_convs_mfff = ConvNormLayer(hidden_dim, hidden_dim, 1, 1, act=act)
        self.fpn_blocks_mfff = CSPRepLayer(hidden_dim*3, hidden_dim, round(3*depth_mult), act=act, expansion=expansion)
        self.upsample_conv_mff = DySample(hidden_dim, scale=2, style="lp")

        self.SAC = SemanticAlignmenCalibration(in_channels_ccff)
        self.focus = Focus(hidden_dim, hidden_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride, self.eval_spatial_size[0] // stride,
                    self.hidden_dim, self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pos_embed)
                # self.register_buffer(f'pos_embed{idx}', pos_embed)
    
    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        '''
        '''
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def forward(self, 
                query,
                key,
                value, 
                ccff_feats,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                **kwargs):
        
        for layer in self.layers:
            memory = layer(
                query,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                **kwargs)
            
        #print("memory", memory.permute(1, 2, 0).shape)

        #for i in range(len(ccff_feats)):
        #    print(f"ccff {i}", ccff_feats[i].shape)
        
        # ! ini menggunakan cara 1, baca: catatan. 
        # Ubah formasi dari [num_queries, bs, embed_dims] ke [bs, C, H, W] sehingga jadi tensor gambar
        h, w = ccff_feats[self.use_encoder_idx[0]].shape[2:]
        ccff_feats = list(ccff_feats)
        ccff_feats[self.use_encoder_idx[0]] = memory.permute(1, 2, 0).reshape(-1, self.hidden_dim, h, w).contiguous()

        for i in range(len(ccff_feats)):
            print(f"ccff {i}", ccff_feats[i].shape)

        # broadcasting and fusion
        inner_outs = [ccff_feats[-1]]
        for idx in range(len(self.in_channels_ccff) - 1, 2, -1):
            feat_high = inner_outs[0]
            feat_low = ccff_feats[idx - 1]
            feat_high = self.lateral_convs[len(self.in_channels_ccff) - 1 - idx](feat_high)
            inner_outs[0] = feat_high
            #print(f"Feat high {idx}", feat_high.shape)
            #print(f"Feat low {idx}", feat_low.shape)
            upsample_feat = self.upsample_conv[len(self.in_channels_ccff) - 1 - idx](feat_high)
            #print(f"upsample_feat {idx}", upsample_feat.shape)
            inner_out = self.fpn_blocks[len(self.in_channels_ccff)-1-idx](torch.concat([upsample_feat, feat_low], dim=1))
            inner_outs.insert(0, inner_out)
        
        #MFF
        feat_high = inner_outs[0]
        feat_low = ccff_feats[1]
        feat_more_low = ccff_feats[0]
        feat_high = self.lateral_convs_mfff(feat_high)
        inner_outs[0] = feat_high
        upsample_feat = self.upsample_conv_mff(feat_high) # upsample make h,w same as feat low
        downsample_feat = self.focus(feat_more_low) # downsample make h.w tp h/2.w/2 which same as feat low
        #print(upsample_feat.shape,downsample_feat.shape,  feat_low.shape)
        #print(torch.concat([feat_high, downsample_feat, feat_low], dim=1))
        inner_out = self.fpn_blocks_mfff(self.mfff(torch.concat([upsample_feat, downsample_feat, feat_low], dim=1)))
        inner_outs.insert(0, inner_out)

        #for feat in range(len(inner_outs)):
        #    print(f"inner out {feat}", inner_outs[feat].shape)


        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels_ccff) - 2):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](torch.concat([downsample_feat, feat_high], dim=1))
            outs.append(out)
        
        #SAC 
        feat_low = outs[-1]
        feat_high = inner_outs[-1]
        out = self.SAC([feat_high, feat_low])
        outs[-1] = out

        #for feat in range(len(outs)):
        #    print(f"outs out {feat}", outs[feat].shape)

        return self._get_encoder_input(outs)
    
    def _get_encoder_input(self, feats):
        # get projection features
        proj_feats = feats
    
        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0, ]
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
            # [l], start index of each level
            level_start_index.append(h * w + level_start_index[-1])

        # [b, l, c]
        feat_flatten = torch.concat(feat_flatten, 1)
        level_start_index.pop()
        return (feat_flatten, spatial_shapes, level_start_index)  # (feat_flatten, spatial_shapes, level_start_index)
