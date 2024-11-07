from typing import Optional

from .decoder import Decoder
from .decoder_perceptron import DecoderPerceptron, DecoderPerceptronCfg

DECODERS = {"perceptron": DecoderPerceptron}
DecoderCfg = DecoderPerceptronCfg

def get_decoder(cfg: DecoderCfg
                ) -> Decoder:
    return DECODERS[cfg.name](cfg)