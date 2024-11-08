import torch
import torch.nn as nn
import torch.utils.checkpoint
from functools import partial
from dataclasses import dataclass
from typing import Literal, Optional
from .encoder import Encoder
from typing import Callable
from .blocks import PositionalEmbedding, SwiGLUFFNFused, Attention, MemEffAttention, NestedTensorBlock as Block

@dataclass
class EncoderTransformerCfg:
    name: Literal["transformer"]
    input_size: Optional[int]
    embed_dim: int
    depth: int
    num_heads: int
    mlp_ratio: float
    qkv_bias: bool
    ffn_bias: bool
    proj_bias: bool
    drop_path_rate: float
    drop_path_uniform: bool
    use_memory_efficient: bool
    init_values: float

class EncoderTransformer(Encoder[EncoderTransformerCfg]):
    def __init__(
        self, cfg: EncoderTransformerCfg):
        super().__init__(cfg)

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.num_features = self.embed_dim = self.cfg.embed_dim  
        self.n_blocks = self.cfg.depth
        self.num_heads = self.cfg.num_heads

        self.input_projection = nn.Linear(in_features=self.cfg.input_size, out_features=self.cfg.embed_dim)
        self.positional_embedding = PositionalEmbedding(demb=self.cfg.embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.cfg.embed_dim))
        
        if self.cfg.drop_path_uniform is True:
            dpr = [self.cfg.drop_path_rate] * self.cfg.depth
        else:
            dpr = [x.item() for x in torch.linspace(0, self.cfg.drop_path_rate, self.cfg.depth)]  # stochastic depth decay rule

        blocks_list = [
            Block(
                dim=self.cfg.embed_dim,
                num_heads=self.cfg.num_heads,
                mlp_ratio=self.cfg.mlp_ratio,
                qkv_bias=self.cfg.qkv_bias,
                proj_bias=self.cfg.proj_bias,
                ffn_bias=self.cfg.ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=nn.GELU,
                ffn_layer=SwiGLUFFNFused,
                init_values=self.cfg.init_values,
                attn_class=MemEffAttention if self.cfg.use_memory_efficient else Attention)
            for i in range(self.cfg.depth)]

        self.blocks = nn.ModuleList(blocks_list)
        self.norm = norm_layer(self.cfg.embed_dim)
        self.head = nn.Identity()
        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.cls_token, std=1e-6)

    def forward(self, features):

        b, s, dim = features.shape
        pos_seq = torch.arange(s-1, -1, -1.0)
        positional_embeds = self.positional_embedding(pos_seq)
        features = self.input_projection(features) + positional_embeds
        features = torch.cat((self.cls_token.expand(features.shape[0], -1, -1), features), dim=1)

        for blk in self.blocks: features = blk(features)
        return features[:, 0] # # # returning CLS token
    
    def feature_dim(self):
        return self.cfg.embed_dim

def named_apply(fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module