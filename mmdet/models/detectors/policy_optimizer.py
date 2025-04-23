

import torch
from mmengine.model import BaseModel
from torch import Tensor
from mmdet.registry import MODELS



@MODELS.register_module()
class DetectorOptimizer(BaseModel):