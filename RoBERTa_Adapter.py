import logging
from typing import Optional, List, Tuple, Union
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.models.roberta.modeling_roberta import (
    RobertaConfig,
    RobertaModel,
    RobertaLayer,
    RobertaEncoder,
    RobertaAttention,
    RobertaIntermediate,
    RobertaOutput,
    RobertaSelfOutput,
    RobertaForSequenceClassification
)

class RobertaWithAdapterConfig(RobertaConfig):
    def __init__(
        self,
        vocab_size=50265,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        adapter_bottleneck_dim = 32,
        adapter_act = "gelu",
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            position_embedding_type=position_embedding_type,
            use_cache=use_cache,
            classifier_dropout=classifier_dropout,
            **kwargs,
        )
        self.adapter_bottleneck_dim = adapter_bottleneck_dim
        self.adapter_act = adapter_act

class Adapter(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.down_proj = nn.Linear(config.hidden_size, config.adapter_bottleneck_dim, bias=True)
        self.act = ACT2FN[config.adapter_act]
        self.up_proj = nn.Linear(config.adapter_bottleneck_dim, config.hidden_size, bias=True)
        
        #near-identity initialization 
        nn.init.trunc_normal_(self.down_proj.weight, mean=0, std=0.01, a=-0.02, b=0.02)
        nn.init.trunc_normal_(self.up_proj.weight, mean=0, std=0.01, a=-0.02, b=0.02)
    def forward(
        self,
        hidden_states: torch.Tensor       
    ):
        input_hidden_states = hidden_states
        hidden_states = self.down_proj(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.up_proj(hidden_states)
        hidden_states = hidden_states + input_hidden_states
        
        return hidden_states

class RobertaSelfOutputWithAdapter(RobertaSelfOutput):
    def __init__(self, config):
        super().__init__(config)
        self.adapter = Adapter(config)

        self.dense.requires_grad_(False)
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.adapter(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        
        return hidden_states

class RobertaAttentionWithAdapter(RobertaAttention):
    def __init__(self, config, position_embedding_type=None):
        super().__init__(config, position_embedding_type)
        self.output = RobertaSelfOutputWithAdapter(config)
        
        self.self.requires_grad_(False)

class RobertaOutputWithAdapter(RobertaOutput):
    def __init__(self, config):
        super().__init__(config)
        self.adapter = Adapter(config)
        
        self.dense.requires_grad_(False)
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.adapter(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
    
class RobertaLayerWithAdapter(RobertaLayer):
    def __init__(self, config):
        super().__init__(config)
        self.attention = RobertaAttentionWithAdapter(config)
        if self.add_cross_attention:
            self.crossattention.requires_grad_(False)
        self.intermediate = RobertaIntermediate(config)
        self.output = RobertaOutputWithAdapter(config)
        
        self.intermediate.requires_grad_(False)

class RobertaEncoderWithAdapter(RobertaEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.ModuleList([RobertaLayerWithAdapter(config) for _ in range(config.num_hidden_layers)])

class RobertaModelWithAdapter(RobertaModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        self.encoder = RobertaEncoderWithAdapter(config)
        self.embeddings.requires_grad_(False)

class RobertaForSeqClsWithAdapter(RobertaForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModelWithAdapter(config)
