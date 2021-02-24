# -*- coding: utf-8 -*-
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config.defaults import get_cfg
from .gtad import GTAD

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    else:
        raise RuntimeError("activation should be relu/gelu, not %s." % activation)


class TransformerEncoder(nn.Module):
    """TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """

    def __init__(self, cfg, drop_out=0.1, activation='relu', norm=None):
        super(TransformerEncoder, self).__init__()
        num_features = cfg.MODEL.FEAT_DIM
        dim_feedforward = cfg.MODEL.TRANSFORMER_DIM
        num_heads = cfg.MODEL.ATTENTION_HEADS
        num_layers = cfg.MODEL.ATTENTION_LAYERS
        encoder_layer = TransformerEncoderLayer(num_features, num_heads, dim_feedforward, drop_out, activation)
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, key_padding_mask=None):
        """Pass the input through the encoder layers in turn.

        Args:
            src: the sequnce to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        for i in range(self.num_layers):
            output = self.layers[i](output, src_mask=mask,
                                    key_padding_mask=key_padding_mask)

        if self.norm:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):
    """TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, src, src_mask=None, key_padding_mask=None):
        """Pass the input through the encoder layer.

        Args:
            src: the sequnce to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=key_padding_mask)[0]
        if key_padding_mask is not None:
            src2 = src2.masked_fill(key_padding_mask.permute(1, 0).unsqueeze(-1), 0)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        if hasattr(self, "activation"):
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        else:  # for backward compatibility
            src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        if key_padding_mask is not None:
            src = src.masked_fill(key_padding_mask.permute(1, 0).unsqueeze(-1), 0)
        return src


class EventDetection(nn.Module):
    def __init__(self, cfg):
        super(EventDetection, self).__init__()
        self.use_env_linear = cfg.MODEL.ENV_HIDDEN_DIM is not None
        self.use_agent_linear = cfg.MODEL.AGENT_HIDDEN_DIM is not None

        if self.use_env_linear:
            self.env_linear = nn.Linear(cfg.MODEL.ENV_DIM, cfg.MODEL.ENV_HIDDEN_DIM)
        if self.use_agent_linear:
            self.agent_linear = nn.Linear(cfg.MODEL.AGENT_DIM, cfg.MODEL.AGENT_HIDDEN_DIM)

        self.agents_fuser = TransformerEncoder(cfg)
        self.agents_environment_fuser = TransformerEncoder(cfg)
        self.event_detector = GTAD(cfg)

        self.attention_steps = cfg.TRAIN.ATTENTION_STEPS

    def forward(self, env_features=None, agent_features=None, agent_masks=None):
        if self.use_env_linear:
            env_features = self.env_linear(env_features)
        if self.use_agent_linear:
            agent_features = self.agent_linear(agent_features)

        if agent_features is None:
            return self.event_detector(env_features.permute(0, 2, 1))

        bsz, tmprl_sz, n_boxes, ft_sz = agent_features.size()
        step = self.attention_steps

        # Fuse all agents together at every temporal point
        smpl_bgn = 0
        agent_fused_features = torch.zeros(bsz, tmprl_sz, ft_sz).cuda()

        if n_boxes > 0:
            for smpl_bgn in range(0, tmprl_sz, step):
                smpl_end = smpl_bgn + step

                fuser_input = agent_features[:, smpl_bgn:smpl_end].contiguous()
                fuser_input = fuser_input.view(-1, n_boxes, ft_sz).permute(1, 0, 2)
                attention_padding_masks = agent_masks[:, smpl_bgn:smpl_end]
                attention_padding_masks = attention_padding_masks.contiguous().view(-1, n_boxes)

                keep_mask = (torch.sum(~attention_padding_masks, dim=-1) > 0)
                keep_indices = torch.masked_select(torch.arange(attention_padding_masks.size(0)).cuda(), keep_mask)

                if len(keep_indices) > 0:
                    fuser_input = fuser_input[:, keep_indices]
                    attention_padding_masks = attention_padding_masks[keep_indices]

                    padded_output = torch.zeros(bsz * (smpl_end - smpl_bgn), ft_sz).cuda()
                    fuser_output = self.agents_fuser(fuser_input, key_padding_mask=attention_padding_masks)
                    fuser_output = torch.sum(fuser_output, dim=0) / torch.sum(~attention_padding_masks, dim=-1, keepdim=True)
                    padded_output[keep_indices] = fuser_output
                    agent_fused_features[:, smpl_bgn:smpl_end] = padded_output.view(bsz, -1, ft_sz)

        if env_features is None:
            return self.event_detector(agent_fused_features.permute(0, 2, 1))
        # fixed_fusion_context = agent_fused_features * 0.4 + env_features * 0.6
        # return self.event_detector(fixed_fusion_context.permute(0, 2, 1))

        env_agent_cat_features = torch.stack([env_features, agent_fused_features], dim=2)

        smpl_bgn = 0
        context_features = torch.zeros(bsz, tmprl_sz, ft_sz).cuda()

        for smpl_bgn in range(0, tmprl_sz, step):
            smpl_end = smpl_bgn + step

            fuser_input = env_agent_cat_features[:, smpl_bgn:smpl_end].contiguous()
            fuser_input = fuser_input.view(-1, 2, ft_sz).permute(1, 0, 2)

            fuser_output = self.agents_environment_fuser(fuser_input)
            fuser_output = torch.mean(fuser_output, dim=0)
            context_features[:, smpl_bgn:smpl_end] = fuser_output.view(bsz, -1, ft_sz)

        return self.event_detector(context_features.permute(0, 2, 1))


if __name__ == '__main__':
    cfg = get_cfg()
    model = EventDetection(cfg)

    batch_size = 1
    temporal_dim = 100
    box_dim = 4
    feature_dim = 2304

    env_input = torch.randn(batch_size, temporal_dim, feature_dim)
    agent_input = torch.randn(batch_size, temporal_dim, box_dim, feature_dim)
    agent_padding_mask = torch.tensor(np.random.randint(0, 1, (batch_size, temporal_dim, box_dim))).bool()

    a, b, c = model(env_input, agent_input, agent_padding_mask)
    print(a.shape, b.shape, c.shape)
