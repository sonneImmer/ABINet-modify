import logging

from fastai.vision import *
import numpy as np

from modules.attention import *
from modules.backbone import ResTransformer_num
from modules.embedding_head import Embedding
from modules.model import Model
from modules.resnet import resnet45, resnet45_num
from .model_vision import BaseVision
from transformers import BertTokenizer, BertModel
from utils import MyDataParallel

class AlignModel(Model):
    def __init__(self, config):
        super().__init__(config)
        self.resnet = resnet45_num()
        self.loss_weight = ifnone(config.model_vision_loss_weight, 1.0)
        self.out_channels = ifnone(config.model_vision_d_model, 512)
        self.vision = BaseVision(config)

        self.is_train = True

        if config.model_vision_attention == 'position':
            mode = ifnone(config.model_vision_attention_mode, 'nearest')
            self.attention5 = PositionAttentionBG(
                max_length=config.dataset_max_length + 1,  # additional stop token
                mode=mode,
                init_with_embedding=False  # should be set to False before v1.1
            )
            self.attention4 = PositionAttentionBG(
                max_length=config.dataset_max_length + 1,  # additional stop token
                mode=mode,
                init_with_embedding=True,  # should be set to False before v1.1
                in_channels=256
            )
        else:
            raise Exception(f'{config.model_vision_attention} is not valid.')
        self.cls = nn.Linear(self.out_channels, self.charset.num_classes)

    def forward(self, images, y):
        
        if self.is_train:
            text_embed = y[-1]
        else:
            v_res = self.vision(images)  # image [67, 3, 32, 128]
            logits = v_res['logits']  # (N, T, C)  # [n, 26, 37] # [67, 26, 7935]
            pt_lengths = self._get_length(logits)
            pt_text, pt_scores, pt_lengths_ = self.decode(logits)
    
            text_embed = []
            for text in pt_text:
                text = self.tokenizer.tokenize(text)
                text_id = self.tokenizer.convert_tokens_to_ids(text) # convert tokens to index
                text_id.insert(0, 101) # add CLS
                text_id.append(102) # add SEP
                text_id = torch.tensor(text_id,dtype = torch.long)
                text_id = text_id.unsqueeze(dim=0)
                text_embedding = self.bert(text_id.cuda())[1][0]       # 取第1层，也可以取别的层。
                text_embedding = text_embedding.detach()   # 切断反向传播
                text_embed.append(text_embedding)
            # print(text_embedding.shape)                # torch.Size([1, 8, 768])
            
            text_embed = torch.stack(text_embed, dim=0)
        
        #fix visual feature
        features = self.resnet(images, layer_num=4) # feature (N, C, H, W) [67, 512, 8, 32]
        attn_vec, attn_scores = self.attention4.add(features, text_embed)
        features = self.resnet.con(attn_vec, 4)

        # attn_vecs, attn_scores = self.attention5(features)
        v_res = self.vision.feature_forward(features)
        logits = v_res['logits']
        pt_lengths = self._get_length(logits)

        return {'feature': attn_vec, 'logits': logits, 'pt_lengths': pt_lengths,
                'attn_scores': attn_scores, 'loss_weight': self.loss_weight, 'name': 'vision'}
    
    def decode(self, logit):
        """ Greed decode """
        # TODO: test running time and decode on GPU
        out = F.softmax(logit, dim=2)
        pt_text, pt_scores, pt_lengths = [], [], []
        for o in out:
            text = self.charset.get_text(o.argmax(dim=1), padding=False, trim=False)
            text = text.split(self.charset.null_char)[0]  # end at end-token
            pt_text.append(text)
            pt_scores.append(o.max(dim=1)[0])
            pt_lengths.append(min(len(text) + 1, self.max_length))  # one for end-token
        pt_scores = torch.stack(pt_scores)
        pt_lengths = pt_scores.new_tensor(pt_lengths, dtype=torch.long)
        return pt_text, pt_scores, pt_lengths