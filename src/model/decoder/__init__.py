from typing import Optional

from .decoder import Decoder
from .decoder_perceptron import DecoderPerceptron, DecoderPerceptronCfg
from .decoder_mlp import DecoderMLP, DecoderMLPCfg

DECODERS = {"perceptron": DecoderPerceptron, 
            "mlp": DecoderMLP}

DecoderCfg = DecoderPerceptronCfg | DecoderMLPCfg

def get_decoder(cfg: DecoderCfg
                ) -> Decoder:
    return DECODERS[cfg.name](cfg)