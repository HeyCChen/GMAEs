import torch
import torch.nn as nn
from .gin import GIN
from torch_geometric.utils import to_dense_batch


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads, need_weight=False):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

        self.need_weight = need_weight

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            x = x + attn_bias
        x = torch.softmax(x, dim=3)
        if self.need_weight:
            attn_weight = x
        x = torch.where(torch.isnan(x), torch.full_like(x, 0), x)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        if self.need_weight:
            return x, attn_weight
        else:
            return x


class TransformerLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads, need_weight=False):
        super(TransformerLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads, need_weight)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

        self.need_weight = need_weight

    def forward(self, x, attn_bias=None):
        y = self.self_attention_norm(x)
        if self.need_weight:
            y, attn_weight = self.self_attention(y, y, y, attn_bias)
        else:
            y = self.self_attention(y, y, y, attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        if self.need_weight:
            return x, attn_weight
        else:
            return x


class GtLayer(nn.Module):
    def __init__(self, in_dim, dropout_rate=0.1, attention_dropout_rate=0.1, trans_num_heads=8, trans_num_layers=1, need_weight=False):
        super(GtLayer, self).__init__()

        transformer_encoders = [TransformerLayer(in_dim, in_dim, dropout_rate, attention_dropout_rate, trans_num_heads, need_weight)
                                for _ in range(trans_num_layers)]
        self.transformer_encoder_layers = nn.ModuleList(transformer_encoders)
        self.transformer_input_dropout = nn.Dropout(p=0.1)

        self.need_weight = need_weight

    def forward(self, input, mask=None):
        output = self.transformer_input_dropout(input)
        attn_weight_list = []
        mask_list = []
        for enc_layer in self.transformer_encoder_layers:
            if self.need_weight:
                output, attn_weight = enc_layer(output)
                attn_weight_list.append(attn_weight)
                mask_list.append(mask)
            else:
                output = enc_layer(output)
        if self.need_weight:
            return output, attn_weight_list, mask_list
        else:
            return output


class GtDecoderLayer(nn.Module):
    def __init__(self, in_dim, dropout_rate=0.1, attention_dropout_rate=0.1, trans_num_heads=8, dec_num_layers=1):
        super(GtDecoderLayer, self).__init__()

        transformer_decoders = [TransformerLayer(in_dim, in_dim, dropout_rate, attention_dropout_rate, trans_num_heads)
                                for _ in range(dec_num_layers)]
        self.transformer_decoder_layers = nn.ModuleList(transformer_decoders)
        self.transformer_input_dropout = nn.Dropout(p=0.1)

    def forward(self, input):
        output = self.transformer_input_dropout(input)
        for enc_layer in self.transformer_decoder_layers:
            output = enc_layer(output)
        return output


class CrossAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(CrossAttention, self).__init__()

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, y1, y2, attn_bias=None):

        batch_size = y2.size(0)

        k = y1
        v = y1
        q = y2
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            x = x + attn_bias
        x = torch.softmax(x, dim=3)
        x = torch.where(torch.isnan(x), torch.full_like(x, 0), x)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x


class CrossTransformerLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads):
        super(CrossTransformerLayer, self).__init__()

        self.attention_norm = nn.LayerNorm(hidden_size)
        self.cross_attention_norm = nn.LayerNorm(hidden_size)
        self.cross_attention = CrossAttention(
            hidden_size, attention_dropout_rate, num_heads)
        self.attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x1, x2, attn_bias=None):
        y1 = self.attention_norm(x1)
        y2 = self.cross_attention_norm(x2)
        y = self.cross_attention(y1, y2, attn_bias)
        y = self.attention_dropout(y)
        x = x2 + y
        # x = x1 + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x


class CtDecoderLayer(nn.Module):
    def __init__(self, in_dim, dropout_rate=0.1, attention_dropout_rate=0.1, trans_num_heads=8, dec_num_layers=1):
        super(CtDecoderLayer, self).__init__()

        transformer_decoders = [CrossTransformerLayer(in_dim, in_dim, dropout_rate, attention_dropout_rate, trans_num_heads)
                                for _ in range(dec_num_layers)]
        self.transformer_decoder_layers = nn.ModuleList(transformer_decoders)
        self.transformer_input_dropout = nn.Dropout(p=0.1)

    def forward(self, input, mask_input):
        output = self.transformer_input_dropout(input)
        mask_output = self.transformer_input_dropout(mask_input)
        for enc_layer in self.transformer_decoder_layers:
            output = enc_layer(output, mask_output)
        return output


class MixedLayer(nn.Module):
    def __init__(self, in_dim, residual, norm, pe_dim, dropout_rate=0.1, attention_dropout_rate=0.1, trans_num_heads=8, num_layers=3, need_weight=False):
        super(MixedLayer, self).__init__()

        self.num_layers = num_layers

        self.gnn = GIN(
            in_dim=int(in_dim),
            num_hidden=int(in_dim),
            out_dim=int(in_dim),
            num_layers=1,
            dropout=dropout_rate,
            activation="prelu",
            residual=residual,
            norm=norm,
            encoding="encoding",
        )
        self.transformer = TransformerLayer(
            in_dim, in_dim, dropout_rate, attention_dropout_rate, trans_num_heads, need_weight)

        self.input_dropout = nn.Dropout(p=0.1)
        self.in_liner = nn.Linear(in_dim, in_dim-pe_dim)

        self.need_weight = need_weight

    def forward(self, input, edge_index, batch, pe, keep_nodes=None):
        output = input.clone()
        attn_weight_list = []
        mask_list = []
        for _ in range(self.num_layers):
            output = self.gnn(output, edge_index)
            if keep_nodes is not None:
                gnn_out = torch.cat(
                    (self.in_liner(output[keep_nodes]), pe), dim=-1)
            else:
                gnn_out = torch.cat(
                    (self.in_liner(output), pe), dim=-1)
            h, h_mask = to_dense_batch(gnn_out, batch)

            if self.need_weight:
                gt_out, attn_weight = self.transformer(h)
                attn_weight_list.append(attn_weight)
                mask_list.append(h_mask)
            else:
                gt_out = self.transformer(h)

            if keep_nodes is not None:
                output[keep_nodes] = gt_out[h_mask]
            else:
                output = gt_out[h_mask]
        if self.need_weight:
            return output, attn_weight_list, mask_list
        else:
            return output
    
class GroupLayer(nn.Module):
    def __init__(self, in_dim, residual, norm, pe_dim, dropout_rate=0.1, attention_dropout_rate=0.1, trans_num_heads=8, num_layers=3, need_weight=False):
        super(GroupLayer, self).__init__()

        self.num_layers = num_layers

        self.gnn = GIN(
            in_dim=int(in_dim),
            num_hidden=int(in_dim),
            out_dim=int(in_dim),
            num_layers=1,
            dropout=dropout_rate,
            activation="prelu",
            residual=residual,
            norm=norm,
            encoding="encoding",
        )
        self.transformer = TransformerLayer(
            in_dim, in_dim, dropout_rate, attention_dropout_rate, trans_num_heads, need_weight)

        self.input_dropout = nn.Dropout(p=0.1)
        self.in_liner = nn.Linear(in_dim, in_dim-pe_dim)

        self.need_weight = need_weight

    def forward(self, input, edge_index, batch, pe, keep_nodes=None):
        output = input.clone()
        attn_weight_list = []
        mask_list = []
        
        for _ in range(self.num_layers):
            output = self.gnn(output, edge_index)
            if keep_nodes is not None:
                gnn_out = torch.cat(
                    (self.in_liner(output[keep_nodes]), pe), dim=-1)
            else:
                gnn_out = torch.cat(
                    (self.in_liner(output), pe), dim=-1)
            h, h_mask = to_dense_batch(gnn_out, batch)

            if self.need_weight:
                gt_out, attn_weight = self.transformer(h)
                attn_weight_list.append(attn_weight)
                mask_list.append(h_mask)
            else:
                gt_out = self.transformer(h)

            if keep_nodes is not None:
                output[keep_nodes] += gt_out[h_mask]
            else:
                output += gt_out[h_mask]
        if self.need_weight:
            return output, attn_weight_list, mask_list
        else:
            return output


class TransConv(torch.nn.Module):
    def __init__(self, channels: int, heads: int, attn_dropout: float = 0.1):
        super().__init__()
        
        self.attn = torch.nn.MultiheadAttention(
            channels,
            heads,
            dropout=attn_dropout,
            batch_first=True,
        )

    def forward(self, x, batch):
        h, mask = to_dense_batch(x, batch)
        h, attn_weight = self.attn(h, h, h, key_padding_mask=~mask, need_weights=True)
        h = h[mask]
        return h, attn_weight, mask
