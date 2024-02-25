from fastai.vision import *

from .model_alignment import BaseAlignment
from .model_language import BCNLanguage
from .model_vision import BaseVision
from modules.embedding_head import Embedding


class StarteModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.use_alignment = ifnone(config.model_use_alignment, True)
        self.max_length = config.dataset_max_length + 1  # additional stop token
        self.vision = BaseVision(config)
        # self.language = BCNLanguage(config)
        self.embedding_loss_weight = ifnone(config.model_vision_embedding_loss_weight, 1.0)
        if self.use_alignment: self.alignment = BaseAlignment(config)
        
        self.embedding = Embedding(8*32, 512)

    def forward(self, images, *args):
        v_res = self.vision(images)
        v_tokens = torch.softmax(v_res['logits'], dim=-1)
        v_lengths = v_res['pt_lengths'].clamp_(2, self.max_length)  # TODO:move to langauge model

        # l_res = self.language(v_tokens, v_lengths)
        # if not self.use_alignment:
        #     return l_res, v_res
        # l_feature, v_feature = l_res['feature'], v_res['feature']
        
        l_feature = l_res
        
        # a_res = self.alignment(l_feature, v_feature)
        add_features = l_feature + v_feature
        
        res = self.vision()
        
        return l_res, v_res
