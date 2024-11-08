from typing import Optional

from .encoder import Encoder
from .encoder_lstm import EncoderLSTM, EncoderLSTMCfg
from .encoder_gru import EncoderGRU, EncoderGRUCfg
from .encoder_transformer import EncoderTransformer, EncoderTransformerCfg

ENCODERS = {"lstm": EncoderLSTM, 
            "gru": EncoderGRU, 
            "transformer": EncoderTransformer}
EncoderCfg = EncoderLSTMCfg | EncoderGRUCfg | EncoderTransformerCfg

def get_encoder(cfg: EncoderCfg
                ) -> Encoder:
    return ENCODERS[cfg.name](cfg)