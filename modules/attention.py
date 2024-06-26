import torch
import torch.nn as nn
from .transformer import PositionalEncoding

class Attention(nn.Module):
    def __init__(self, in_channels=512, max_length=25, n_feature=256):
        super().__init__()
        self.max_length = max_length

        self.f0_embedding = nn.Embedding(max_length, in_channels)
        self.w0 = nn.Linear(max_length, n_feature)
        self.wv = nn.Linear(in_channels, in_channels)
        self.we = nn.Linear(in_channels, max_length)

        self.active = nn.Tanh()
        self.softmax = nn.Softmax(dim=2)

    def forward(self, enc_output):
        enc_output = enc_output.permute(0, 2, 3, 1).flatten(1, 2)
        reading_order = torch.arange(self.max_length, dtype=torch.long, device=enc_output.device)
        reading_order = reading_order.unsqueeze(0).expand(enc_output.size(0), -1)  # (S,) -> (B, S)
        reading_order_embed = self.f0_embedding(reading_order)  # b,25,512

        t = self.w0(reading_order_embed.permute(0, 2, 1))  # b,512,256
        t = self.active(t.permute(0, 2, 1) + self.wv(enc_output))  # b,256,512

        attn = self.we(t)  # b,256,25
        attn = self.softmax(attn.permute(0, 2, 1))  # b,25,256
        g_output = torch.bmm(attn, enc_output)  # b,25,512
        return g_output, attn.view(*attn.shape[:2], 8, 32)


def encoder_layer(in_c, out_c, k=3, s=2, p=1):
    return nn.Sequential(nn.Conv2d(in_c, out_c, k, s, p),
                         nn.BatchNorm2d(out_c),
                         nn.ReLU(True))

def decoder_layer(in_c, out_c, k=3, s=1, p=1, mode='nearest', scale_factor=None, size=None):
    align_corners = None if mode=='nearest' else True
    return nn.Sequential(nn.Upsample(size=size, scale_factor=scale_factor, 
                                     mode=mode, align_corners=align_corners),
                         nn.Conv2d(in_c, out_c, k, s, p),
                         nn.BatchNorm2d(out_c),
                         nn.ReLU(True))


class PositionAttention(nn.Module):
    def __init__(self, max_length, in_channels=512, num_channels=64, 
                 h=8, w=32, mode='nearest', init_with_embedding=False, **kwargs):
        super().__init__()
        self.max_length = max_length
        self.init_with_embedding = init_with_embedding
        self.k_encoder = nn.Sequential(
            encoder_layer(in_channels, num_channels, s=(1, 2)),
            encoder_layer(num_channels, num_channels, s=(2, 2)),
            encoder_layer(num_channels, num_channels, s=(2, 2)),
            encoder_layer(num_channels, num_channels, s=(2, 2))
        )
        self.k_decoder = nn.Sequential(
            decoder_layer(num_channels, num_channels, scale_factor=2, mode=mode),
            decoder_layer(num_channels, num_channels, scale_factor=2, mode=mode),
            decoder_layer(num_channels, num_channels, scale_factor=2, mode=mode),
            decoder_layer(num_channels, in_channels, size=(h, w), mode=mode)
        )

        self.pos_encoder = PositionalEncoding(in_channels, dropout=0, max_len=max_length)
        self.project = nn.Linear(in_channels, in_channels)
        
        # Fix: `embedding_func` works only when use embedding init_state in v1.x,
        # so when should limit `embedding_func` by the `init_with_embedding` signal.
        if self.init_with_embedding:
            self.embedding_func = nn.Linear(300, 512)

    def forward(self, x, embedding_vector=None):
        N, E, H, W = x.size() # [67, 512, 8, 32]
        k, v = x, x  # (N, E, H, W)

        # calculate key vector
        features = []
        for i in range(0, len(self.k_encoder)):
            k = self.k_encoder[i](k)
            features.append(k)
        for i in range(0, len(self.k_decoder) - 1):
            k = self.k_decoder[i](k)
            k = k + features[len(self.k_decoder) - 2 - i]
        k = self.k_decoder[-1](k) # [67, 512, 8, 32]

        # calculate query vector
        # TODO q=f(q,k)
        if self.init_with_embedding:
            init_state = self.embedding_func(embedding_vector)  # [450, 512]
            init_state = init_state.repeat(self.max_length, 1, 1)  # [26, 450, 512]
        else:
            init_state = x.new_zeros((self.max_length, N, E))  # (T, N, E)  # [26, 450, 512]
        q = self.pos_encoder(init_state)  # (T, N, E)
        q = q.permute(1, 0, 2)  # (N, T, E)
        q = self.project(q)  # (N, T, E)
        
        # calculate attention
        attn_scores = torch.bmm(q, k.flatten(2, 3))  # (N, T, (H*W))
        attn_scores = attn_scores / (E ** 0.5)
        attn_scores = torch.softmax(attn_scores, dim=-1)

        v = v.permute(0, 2, 3, 1).view(N, -1, E)  # (N, (H*W), E)
        attn_vecs = torch.bmm(attn_scores, v)  # (N, T, E)

        return attn_vecs, attn_scores.view(N, -1, H, W)
    
class PositionAttentionBG(nn.Module):
    def __init__(self, max_length, in_channels=512, num_channels=64, 
                 h=8, w=32, mode='nearest', init_with_embedding=False, **kwargs):
        super().__init__()
        self.max_length = max_length
        self.init_with_embedding = init_with_embedding
        self.k_encoder = nn.Sequential(
            encoder_layer(in_channels, num_channels, s=(1, 2)),
            encoder_layer(num_channels, num_channels, s=(2, 2)),
            encoder_layer(num_channels, num_channels, s=(2, 2)),
            encoder_layer(num_channels, num_channels, s=(2, 2))
        )
        self.k_decoder = nn.Sequential(
            decoder_layer(num_channels, num_channels, scale_factor=2, mode=mode),
            decoder_layer(num_channels, num_channels, scale_factor=2, mode=mode),
            decoder_layer(num_channels, num_channels, scale_factor=2, mode=mode),
            decoder_layer(num_channels, in_channels, size=(h, w), mode=mode)
        )

        self.pos_encoder = PositionalEncoding(in_channels, dropout=0, max_len=256)
        self.project = nn.Linear(in_channels, in_channels)
        
        # Fix: `embedding_func` works only when use embedding init_state in v1.x,
        # so when should limit `embedding_func` by the `init_with_embedding` signal.
        if self.init_with_embedding:
            self.embedding_func = nn.Linear(768, in_channels)

    def forward(self, x, embedding_vector=None):
        N, E, H, W = x.size()
        k, v = x, x  # (N, E, H, W)

        # calculate key vector
        features = []
        for i in range(0, len(self.k_encoder)):
            k = self.k_encoder[i](k)
            features.append(k)
        for i in range(0, len(self.k_decoder) - 1):
            k = self.k_decoder[i](k)
            k = k + features[len(self.k_decoder) - 2 - i]
        k = self.k_decoder[-1](k)

        # calculate query vector
        # TODO q=f(q,k)
        if self.init_with_embedding:
            init_state = self.embedding_func(embedding_vector)  # embedding_vectoe [66, 768] init_state [66, 512]
            init_state = init_state.repeat(self.max_length, 1, 1)  # [26, 66, 512]
        else:
            init_state = x.new_zeros((self.max_length, N, E))  # (T, N, E)  # [26, 450, 512]
        q = self.pos_encoder(init_state)  # (T, N, E)
        q = q.permute(1, 0, 2)  # (N, T, E)
        q = self.project(q)  # (N, T, E)
        q = q + init_state.permute(1, 0, 2)

        # calculate attention
        attn_scores = torch.bmm(q, k.flatten(2, 3))  # (N, T, (H*W))
        attn_scores = attn_scores / (E ** 0.5)
        attn_scores = torch.softmax(attn_scores, dim=-1)

        v = v.permute(0, 2, 3, 1).view(N, -1, E)  # (N, (H*W), E)
        attn_vecs = torch.bmm(attn_scores, v)  # (N, T, E)

        # back_f = attn_vecs.view(N, E, H, W)

        return attn_vecs, attn_scores.view(N, -1, H, W)


    def add(self, x, embedding_vector=None):
        N, E, H, W = x.size()
        k, v = x, x  # (N, E, H, W)

        # calculate key vector
        features = []
        for i in range(0, len(self.k_encoder)):
            k = self.k_encoder[i](k)
            features.append(k)
        for i in range(0, len(self.k_decoder) - 1):
            k = self.k_decoder[i](k)
            k = k + features[len(self.k_decoder) - 2 - i]
        k = self.k_decoder[-1](k)

        # calculate query vector
        # TODO q=f(q,k)
        if self.init_with_embedding:
            init_state = self.embedding_func(embedding_vector)  # embedding_vectoe [66, 768] init_state [66, 512]
            init_state = init_state.repeat(256, 1, 1)  # [26, 66, 512]
        else:
            init_state = x.new_zeros((self.max_length, N, E))  # (T, N, E)  # [26, 450, 512]
        q = self.pos_encoder(init_state)  # (T, N, E)
        q = q.permute(1, 0, 2)  # (N, T, E)
        q = self.project(q)  # (N, T, E)
        q = q + init_state.permute(1, 0, 2)

        # calculate attention
        attn_scores = torch.bmm(q, k.flatten(2, 3))  # (N, T, (H*W))
        attn_scores = attn_scores / (E ** 0.5)
        attn_scores = torch.softmax(attn_scores, dim=-1)

        v = v.permute(0, 2, 3, 1).view(N, -1, E)  # (N, (H*W), E)
        attn_vecs = torch.bmm(attn_scores, v)  # (N, T, E)

        back_f = attn_vecs.permute(0, 2, 1).view(N, E, H, W)
        back_f = back_f + x

        return back_f, attn_scores

class BilinearSeqAttn(nn.Module):
    def __init__(self,  **kwargs):
        super().__init__()
        self.cls = nn.Linear(768, 768)
    
    def bsa(self, x, embedding_vector=None):
        L, E = x.size()
        k, v = x, x

        k = k.permute(1, 0)

        init_state = embedding_vector.repeat(3, 1)
        q = self.cls(init_state)
        q = q + init_state

        attn_scores = torch.matmul(q, k)
        attn_scores = attn_scores / (E ** 0.5)
        attn_scores = torch.softmax(attn_scores, dim=-1)

        attn_vecs = torch.matmul(attn_scores, v)

        return attn_vecs, attn_scores