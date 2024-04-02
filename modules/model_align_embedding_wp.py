import logging

from fastai.vision import *
import numpy as np

from modules.attention import *
from modules.backbone import ResTransformer_num
from modules.embedding_head import Embedding
from modules.model import Model
from modules.resnet import resnet45, resnet45_num
from .model_vision import BaseVision
from pytorch_pretrained_bert import BertTokenizer, BertModel
# from transformers import BertTokenizer, BertModel
from utils import MyDataParallel
from ltp import LTP

class AlignModel(Model):
    def __init__(self, config):
        super().__init__(config)
        self.resnet = resnet45_num()
        self.loss_weight = ifnone(config.model_vision_loss_weight, 1.0)
        self.out_channels = ifnone(config.model_vision_d_model, 512)
        self.vision = BaseVision(config)
        # self.tokenizer = BertTokenizer.from_pretrained('./workdir/bert-base-chinese/') # 加载base模型的对应的切词器
        # self.bert = BertModel.from_pretrained('./workdir/bert-base-chinese')
        # self.ltp = LTP('./workdir/small').cuda()

        self.is_train = True
        if config.model_vision_backbone == 'transformer':
            self.backbone = ResTransformer_num(config)
        else:
            self.backbone = resnet45()
    
        if config.model_vision_attention == 'position':
            mode = ifnone(config.model_vision_attention_mode, 'nearest')
            self.attention3 = PositionAttentionBG(
                max_length=config.dataset_max_length + 1,  # additional stop token
                mode=mode,
                init_with_embedding=True,  # should be set to False before v1.1
                in_channels=128
            )
            self.attention4 = PositionAttentionBG(
                max_length=config.dataset_max_length + 1,  # additional stop token
                mode=mode,
                init_with_embedding=True,  # should be set to False before v1.1
                in_channels=256
            )
            self.attention5 = PositionAttentionBG(
                max_length=config.dataset_max_length + 1,  # additional stop token
                mode=mode,
                init_with_embedding=True  # should be set to False before v1.1
            )
            self.attentionwp = BilinearSeqAttn(
                max_length=8,  # additional stop token
                mode=mode,
                init_with_embedding=True,  # should be set to False before v1.1
                in_channels=768
            )
        elif config.model_vision_attention == 'attention':
            self.attention = Attention(
                max_length=config.dataset_max_length + 1,  # additional stop token
                n_feature=8 * 32,
            )
        else:
            raise Exception(f'{config.model_vision_attention} is not valid.')
        self.cls = nn.Linear(self.out_channels, self.charset.num_classes)

    def forward(self, images, y):
        
        wp_embedd = y[-1]
        # v_res = self.vision(images)  # image [67, 3, 32, 128]
        # logits = v_res['logits']  # (N, T, C)  # [n, 26, 37] # [67, 26, 7935]
        # pt_lengths = self._get_length(logits)
        # pt_text, pt_scores, pt_lengths_ = self.decode(logits)

        embedding_vec_layer3 = []
        embedding_vec_layer4 = []
        embedding_vec_layer5 = []
        # text_embeddings = []
        # for text in pt_text:
        #     text_t = self.tokenizer.tokenize(text)
        #     text_id = self.tokenizer.convert_tokens_to_ids(text_t) # convert tokens to index
        #     text_id.insert(0, 101) # add CLS, add SEP
        #     text_id.append(102)
        #     text_id = torch.tensor(text_id,dtype = torch.long).unsqueeze(dim=0)
        #     text_embedding = self.bert(text_id.cuda())[1][0]       # 取第1层，也可以取别的层。
        #     text_embedding = text_embedding.detach()   # 切断反向传播
        #     text_embeddings.append(text_embedding)

        #     # wordpiece
        #     word_embeddings = []
        #     wp = self.ltp.pipeline([text], tasks=["cws"])
        #     if len(wp) == 0:
        #         word_ids = self.tokenizer.tokenize(w)
        #         word_ids = self.tokenizer.convert_tokens_to_ids(word_ids)
        #         word_ids.insert(0, 101) # add CLS
        #         word_ids.append(102)
        #         word_ids = torch.tensor(word_ids).unsqueeze(0)
        #         word_embedding = self.bert(word_ids.cuda())[1][0]
        #         word_embedding = word_embedding.detach()
        #         word_embeddings.append(word_embedding)
        #     else:
        #         for w in wp[0][0]:
        #             word_ids = self.tokenizer.tokenize(w)
        #             word_ids = self.tokenizer.convert_tokens_to_ids(word_ids)
        #             word_ids.insert(0, 101) # add CLS    
        #             word_ids.append(102)
        #             word_ids = torch.tensor(word_ids).unsqueeze(0)
        #             word_embedding = self.bert(word_ids.cuda())[1][0]
        #             word_embedding = word_embedding.detach()
        #             word_embeddings.append(word_embedding)
            
        #     word_embeddings = torch.stack(word_embeddings, dim=0)
        #     attn_vec_word, _ = self.attentionwp.bsa(word_embeddings, text_embedding)

        embedding_vec_layer3.append(wp_embedd[0])
        embedding_vec_layer4.append(wp_embedd[1])
        embedding_vec_layer5.append(wp_embedd[2])

        embedding_vec_layer3 = torch.stack(embedding_vec_layer3, dim=0)
        embedding_vec_layer4 = torch.stack(embedding_vec_layer4, dim=0)
        embedding_vec_layer5 = torch.stack(embedding_vec_layer5, dim=0)

        #fix visual feature
        #3
        features = self.resnet(images, layer_num=3) # feature (N, C, H, W) [67, 512, 8, 32]
        attn_vec, attn_scores = self.attention3.add(features, embedding_vec_layer3)
        features = self.resnet.con3(attn_vec, 3)
        #4
        # features = self.resnet(images, layer_num=4) # feature (N, C, H, W) [67, 512, 8, 32]
        attn_vec, attn_scores = self.attention4.add(features, embedding_vec_layer4)
        features = self.resnet.con3(attn_vec, 4)
        #5
        attn_vec, attn_scores = self.attention5(features, embedding_vec_layer5)

        logits = self.cls(attn_vec)
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