__all__ = ['ConvTimeNet']

# Cell
import torch
from torch import nn
from TSForecasting.layers.ConvTimeNet_backbone import ConvTimeNet_backbone


# ConvTimeNet: depatch + batch norm + gelu + Conv + 2-layer-ffn(PointWise Conv + PointWise Conv)
class Model(nn.Module):
    def __init__(self, enc_in, seq_len, pred_len, e_layers, d_model, d_ff, dropout,
                 head_dropout, patch_ks, patch_sd, padding_patch, revin, affine,
                 subtract_last, dw_ks, re_param, re_param_kernel, enable_res_param,
                 norm: str = 'batch', act: str = "gelu", head_type='flatten'):

        super().__init__()

        # load parameters
        c_in = enc_in
        context_window = seq_len
        target_window = pred_len

        n_layers = e_layers
        d_model = d_model
        d_ff = d_ff
        dropout = dropout
        head_dropout = head_dropout

        patch_len = patch_ks
        stride = patch_sd
        padding_patch = padding_patch

        revin = revin
        affine = affine
        subtract_last = subtract_last

        seq_len = seq_len
        dw_ks = dw_ks

        re_param = re_param
        re_param_kernel = re_param_kernel
        enable_res_param = enable_res_param

        # model
        self.model = ConvTimeNet_backbone(c_in=c_in, seq_len=seq_len, context_window=context_window,
                                          target_window=target_window, patch_len=patch_len, stride=stride,
                                          n_layers=n_layers, d_model=d_model, d_ff=d_ff, dw_ks=dw_ks, norm=norm,
                                          dropout=dropout, act=act, head_dropout=head_dropout,
                                          padding_patch=padding_patch, head_type=head_type,
                                          revin=revin, affine=affine, deformable=True, subtract_last=subtract_last,
                                          enable_res_param=enable_res_param, re_param=re_param,
                                          re_param_kernel=re_param_kernel)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.model(x)
        x = x.permute(0, 2, 1)

        return x
