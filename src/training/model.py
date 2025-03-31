import torch
import torch.nn as nn
import torch.nn.functional as thf
from asteroid_filterbanks import make_enc_dec
from torch.nn.parameter import Parameter

import torch.optim as optim
from torchmetrics.functional import (
    scale_invariant_signal_noise_ratio as si_snr,
    signal_noise_ratio as snr,
    signal_distortion_ratio as sdr,
    scale_invariant_signal_distortion_ratio as si_sdr)


# from torch.nn.utils import weight_norm

class ACNTCN(nn.Module):

    def __init__(
            self,
            ch_dim,
            label_emb_dim=512,
            n_fft=256,
            stride=128,
            window="hann",
            n_layers=6,
            eps=1.0e-5,
    ):
        super().__init__()
        self.n_srcs = 1
        self.n_layers = n_layers
        self.n_imics = 1
        
        assert n_fft % 2 == 0
        n_freqs = n_fft // 2 + 1
        self.n_freqs = n_freqs
        self.eps = eps

        self.chunk_size = stride

        self.istft_pad = n_fft - stride

        # ISTFT overlap-add will affect this many chunks in the future
        self.istft_lookback = 1 + (self.istft_pad - 1) // self.istft_pad

        self.enc, self.dec = make_enc_dec('stft', n_filters=n_fft,
                                          kernel_size=n_fft,
                                          stride=stride,
                                          window_type=window)

        self.t_ksize = 3
        self.ch_dim = ch_dim

        self.half_freq_dim = n_freqs // 2

        self.in_conv = nn.Sequential(
            nn.Conv2d(self.n_imics * 2, ch_dim, kernel_size=(3, 3), padding=(0, 1)),
        )
        
        self.mask_net = MaskNet(ch_dim, n_freqs, n_layers, 4, 8)

        self.label_proj = nn.Sequential(
            nn.Linear(label_emb_dim, self.ch_dim * self.half_freq_dim),
            nn.LayerNorm(self.ch_dim * self.half_freq_dim),
        )

        self.ou_conv = nn.Sequential(
            nn.Conv2d(ch_dim, self.n_imics * 2, kernel_size=(3, 3), padding=(0, 1)),
        )

    def init_buffers(self, batch_size, device):
        conv_buf = torch.zeros(batch_size, self.n_imics * 2, self.t_ksize - 1, self.n_freqs,
                               device=device)
        istft_buf = torch.zeros(batch_size, self.n_srcs, self.n_freqs * 2, self.istft_lookback,
                                device=device)
        deconv_buf = torch.zeros(batch_size, self.ch_dim, self.t_ksize - 1, self.n_freqs,
                                 device=device)
        return dict(conv_buf=conv_buf, deconv_buf=deconv_buf, istft_buf=istft_buf)

    def forward(
            self,
            x: torch.Tensor,
            label_embedding: torch.Tensor,
            input_state=None
    ):
        if input_state is None:
            input_state = self.init_buffers(x.shape[0], x.device)

        conv_buf = input_state['conv_buf']
        deconv_buf = input_state['deconv_buf']
        istft_buf = input_state['istft_buf']

        batch = self.enc(x)  # [B, M, nfft + 2, T]

        batch = torch.stack((batch[..., :self.n_freqs, :], batch[..., self.n_freqs:, :]), dim=1)  # [B, 2*M, tf, T]
        batch = batch.transpose(2, 3).contiguous()  # [B, M, T, tf]

        batch = torch.cat((conv_buf, batch), dim=2)
        conv_buf = batch[:, :, -(self.t_ksize - 1):, :]

        batch = self.in_conv(batch)

        B, M, T, C = batch.shape

        label = self.label_proj(label_embedding)
        label = label.reshape([B, self.ch_dim, self.half_freq_dim]).unsqueeze(2)
        mask = self.mask_net(batch, label)
        batch = mask * batch

        batch = torch.cat((deconv_buf, batch), dim=2)
        deconv_buf = batch[:, :, -(self.t_ksize - 1):, :]
        batch = self.ou_conv(batch)

        batch = batch.view([B, self.n_srcs, 2, T, C]).transpose(3, 4)
        batch = torch.cat([batch[:, :, 0], batch[:, :, 1]], dim=2)

        batch = torch.cat([istft_buf, batch], dim=3)
        istft_buf = batch[..., -self.istft_lookback:]

        batch = self.dec(batch)  # [B, n_srcs, n_srcs, -1]
        batch = batch[..., self.istft_lookback * self.chunk_size:]
        input_state['conv_buf'] = conv_buf
        input_state['deconv_buf'] = deconv_buf
        input_state['istft_buf'] = istft_buf
        return batch, input_state


class LayerNormalizationC(nn.Module):
    def __init__(self, C, eps=1e-5, preserve_outdim=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(C))
        self.bias = nn.Parameter(torch.zeros(C))
        self.eps = eps
        self.normalized_shape = (C,)

    def forward(self, x: torch.Tensor):
        """
        est: (*, C)
        """
        x = x.transpose(1, 3)
        x = thf.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.transpose(1, 3).contiguous()
        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_channel, out_channel, n_layer, r, n_head, freq_dim, kernel_size, stride):
        super(EncoderBlock, self).__init__()
        self.tfcm = TFCM(in_channel, n_layer, r)
        self.attn = ConvSelfAttn(in_channel, in_channel // n_head, freq_dim, n_head)
        self.freq_down = FreqDown(in_channel, out_channel, (1, kernel_size), (1, stride))

    def forward(self, x):
        x = self.tfcm(x)
        x = self.attn(x)
        x = self.freq_down(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channel, out_channel, freq_dim, n_layer, r, n_head, kernel_size, stride):
        super(DecoderBlock, self).__init__()
        self.fe = FeatureEnh(in_channel, freq_dim)
        self.freq_up = FreqUp(in_channel, out_channel, (1, kernel_size), (1, stride))
        self.tfcm = TFCM(out_channel, n_layer, r)
        self.attn = ConvSelfAttn(out_channel, out_channel // n_head, freq_dim * 2 + 1, n_head)

    def forward(self, x1, x2):
        x = self.fe(x1, x2)
        x = self.freq_up(x)
        x = self.tfcm(x)
        x = self.attn(x)
        return x


class MiddleBlock(nn.Module):
    def __init__(self, channel, n_layer, freq_dim, n_head, r):
        super(MiddleBlock, self).__init__()
        self.tfcm = TFCM(channel, n_layer, r)
        self.attn = ConvSelfAttn(channel, channel // n_head, freq_dim, n_head)

    def forward(self, x):
        x = self.tfcm(x)
        x = self.attn(x)
        return x


class MaskNet(nn.Module):
    def __init__(self, ch_dim, freq_dim, n_layer, r, n_head):
        super().__init__()

        self.encoder = EncoderBlock(ch_dim, ch_dim, n_layer, r, n_head, freq_dim, 3, 2)

        self.middle = nn.Sequential(
            MiddleBlock(ch_dim, n_layer, freq_dim // 2, n_head, r),
            MiddleBlock(ch_dim, n_layer, freq_dim // 2, n_head, r),
            MiddleBlock(ch_dim, n_layer, freq_dim // 2, n_head, r),
        )

        self.decoder = DecoderBlock(ch_dim, ch_dim, freq_dim // 2, n_layer, r, n_head, 3, 2)

    def forward(self, x, label):
        xc = self.encoder(x)

        x = self.middle(xc)

        x = x + label

        x = self.decoder(x, xc)
        return x


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        return thf.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)


class TfcmBlock(nn.Module):
    def __init__(self, channel, dilation, r=2):
        super().__init__()
        t_width = 2
        time_pad = t_width + (dilation - 1) * (t_width - 1) - 1
        freq_pad = 3
        self.pad = nn.ConstantPad2d((freq_pad, freq_pad, time_pad, 0), value=0.0)
        self.dwconv = nn.Conv2d(channel, channel, kernel_size=(t_width, 7), dilation=(dilation, 1),
                                groups=channel) 
        self.norm = LayerNorm(channel, eps=1e-6)
        self.pwconv1 = nn.Linear(channel, r * channel)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(r * channel, channel)

    def forward(self, x):
        inp = x
        x = self.pad(x)
        x = self.dwconv(x)
        # x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = inp + x
        return x


class LayerNormalization4DCF(nn.Module):
    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        assert len(input_dimension) == 2
        self.eps = eps

        param_size = [1, input_dimension[0], 1, input_dimension[1]]
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))

        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)

    def forward(self, x):
        if x.ndim == 4:
            stat_dim = (1, 3)
        else:
            raise ValueError("Expect est to have 4 dimensions, but got {}".format(x.ndim))

        mu_ = x.mean(dim=stat_dim, keepdim=True)  # [B,1,T,1]
        std_ = torch.sqrt(x.var(dim=stat_dim, unbiased=False, keepdim=True) + self.eps)  # [B,1,T,1]
        x_hat = ((x - mu_) / (std_ + self.eps)) * self.gamma + self.beta

        return x_hat


class ConvSelfAttn(nn.Module):
    def __init__(self, d_model, embed_dim, freq_dim, num_heads):
        super(ConvSelfAttn, self).__init__()
        self.q = nn.Sequential(
            nn.Conv2d(d_model, embed_dim * num_heads, kernel_size=1, bias=False),
            nn.ReLU(),
            LayerNormalization4DCF((embed_dim * num_heads, freq_dim))
        )
        self.k = nn.Sequential(
            nn.Conv2d(d_model, embed_dim * num_heads, kernel_size=1, bias=False),
            nn.ReLU(),
            LayerNormalization4DCF((embed_dim * num_heads, freq_dim))
        )
        self.v = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=1, bias=False),
            nn.ReLU(),
            LayerNormalization4DCF((embed_dim * num_heads, freq_dim))
        )
        self.o = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=1),
            nn.ReLU(),
            LayerNormalization4DCF((embed_dim * num_heads, freq_dim))
        )

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.d_model = d_model
        self.d_v = d_model // num_heads

    def get_mask(self, T, device):
        one_matrix = torch.ones((T, T), device=device)
        mask = torch.triu(one_matrix, diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, x, do_mask=True):
        B, M, T, F = x.shape
        q = self.q(x).view(B, self.num_heads, self.embed_dim, T, F).transpose(2, 3).contiguous().view(B, self.num_heads,
                                                                                                      T, -1)
        k = self.k(x).view(B, self.num_heads, self.embed_dim, T, F).transpose(2, 3).contiguous().view(B, self.num_heads,
                                                                                                      T, -1)
        v = self.v(x).view(B, self.num_heads, self.d_v, T, F).transpose(2, 3).contiguous().view(B, self.num_heads, T,
                                                                                                -1)
        # B H T Q_dim tf
        norm_dim = q.size(-1)
        attn = q @ k.transpose(-1, -2) / (norm_dim ** 0.5)  # B n T T
        if do_mask:
            mask = self.get_mask(T, device=x.device)
            attn = attn.masked_fill(mask == 1, -1e9)
        attn = torch.softmax(attn, dim=-1)
        scores = attn @ v  # B H T freq*dqkv
        scores = scores.view(B, self.num_heads, T, self.d_v, F).transpose(2, 3).contiguous().view(B, -1, T, F)
        scores = self.o(scores)
        return scores + x


class TFCM(nn.Module):
    def __init__(self, channel, num_layers=6, r=4):
        super(TFCM, self).__init__()
        self.layers = nn.ModuleList([])
        for i in range(num_layers):
            self.layers.append(
                TfcmBlock(channel, 2 ** i, r=r)
            )
        # self.norm = LayerNormalizationC(channel)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        # est = self.norm(est)
        return x


class FreqDown(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride):
        super(FreqDown, self).__init__()
        padding = (kernel_size[-1] - stride[-1]) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, (0, padding)),
            nn.PReLU(out_channel),
        )

    def forward(self, x):
        # if self.fc is not None:
        #     est = self.fc(est)
        return self.conv(x)


class FeatureEnh(nn.Module):
    def __init__(self, channel, freq_dim):
        super(FeatureEnh, self).__init__()

        self.weight = nn.Sequential(
            nn.Conv2d(2 * channel, channel, 1),
            nn.Linear(freq_dim, freq_dim // 2),
            nn.ReLU(),
            nn.Linear(freq_dim // 2, freq_dim),
            nn.Sigmoid(),
        )
    
    def forward(self, x1, x2):  # x1: [B Q T C]  x2: [B Q T C]
        x = self.weight(torch.cat([x1, x2], dim=1)) * x1
        return x
    

class FreqUp(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride):
        super(FreqUp, self).__init__()
        padding = (kernel_size[-1] - stride[-1]) // 2

        # self.pw = nn.Sequential(
        #     nn.Conv2d(in_channel, out_channel, 1),
        #     nn.PReLU(out_channel),
        # )
        self.tconv = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size, stride, padding=(0, padding)),
            nn.PReLU(out_channel),
        )

    def forward(self, x):
        # x = self.pw(x)
        x = self.tconv(x)
        return x


def mod_pad(x, chunk_size, pad):
    # Mod pad the x to perform integer number of
    # inferences
    mod = 0
    if (x.shape[-1] % chunk_size) != 0:
        mod = chunk_size - (x.shape[-1] % chunk_size)

    x = thf.pad(x, (0, mod))
    x = thf.pad(x, pad)

    return x, mod


class Net(nn.Module):
    def __init__(self, label_len, ch_dim,  n_fft=256, stride=128, label_emb_dim=512,
                 n_layers=6):
        super(Net, self).__init__()

        self.net = ACNTCN(ch_dim=ch_dim, label_emb_dim=label_emb_dim, n_fft=n_fft, stride=stride, n_layers=n_layers)

        self.label_embedding = nn.Sequential(
            nn.Linear(label_len, label_emb_dim),
            nn.LayerNorm(label_emb_dim),
            nn.ReLU(),
            nn.Linear(label_emb_dim, label_emb_dim),
            nn.LayerNorm(label_emb_dim),
            nn.ReLU(),
            nn.Linear(label_emb_dim, label_emb_dim),
            nn.LayerNorm(label_emb_dim),
            nn.ReLU(),
            nn.Linear(label_emb_dim, label_emb_dim),
            nn.LayerNorm(label_emb_dim),
            nn.ReLU()
        )

    def init_buffers(self, batch_size, device):
        return self.net.init_buffers(batch_size, device)

    def forward(self, x, label, input_state=None, pad=True, writer=None, step=None, idx=None):
        label_embedding = self.label_embedding(label)

        x, _ = self.net(x, label_embedding)

        return x


def optimizer(model, data_parallel=False, **kwargs):
    return optim.Adam(model.parameters(), **kwargs)


def loss(pred, tgt):
    return -0.9 * snr(pred, tgt).mean() - 0.1 * si_snr(pred, tgt).mean()


def metrics(mixed, output, gt):
    """ Function to compute metrics """
    metrics = {}

    def metric_i(metric, src, pred, tgt):
        _vals = []
        for s, t, p in zip(src, tgt, pred):
            _vals.append((metric(p, t) - metric(s, t)).cpu().item())
        return _vals

    for m_fn in [snr, si_snr]:
        metrics[m_fn.__name__] = metric_i(m_fn,
                                          mixed[:, :gt.shape[1], :],
                                          output,
                                          gt)

    return metrics


if __name__ == "__main__":
    model = Net(41, 32)
    x = torch.randn((1, 1, 16000))
    l = torch.randn((1, 41))
    from thop import profile

    ops, params = profile(model, (x, l))
    print(ops / 1e9)
    print(params / 1e6)
