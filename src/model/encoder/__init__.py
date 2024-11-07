from typing import Optional

from .encoder import Encoder
from .encoder_lstm import EncoderLSTM, EncoderLSTMCfg

ENCODERS = {"lstm": EncoderLSTM}
EncoderCfg = EncoderLSTMCfg

def get_encoder(cfg: EncoderCfg
                ) -> Encoder:
    return ENCODERS[cfg.name](cfg)