from .loss import Loss
from .loss_cce import LossCCE, LossCCECfgWrapper

LOSSES = {
    LossCCECfgWrapper: LossCCE}

LossCfgWrapper = LossCCECfgWrapper 

def get_losses(cfgs: list[LossCfgWrapper]) -> list[Loss]:
    return [LOSSES[type(cfg)](cfg) for cfg in cfgs]