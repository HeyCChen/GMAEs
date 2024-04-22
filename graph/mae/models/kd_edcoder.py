from typing import Optional
from functools import partial

import numpy as np

import torch
import torch.nn as nn

from mae.models.transformer import GtLayer, GtDecoderLayer, CtDecoderLayer, MixedLayer, GroupLayer

from .gat import GAT
from .gin import GIN
from .loss_func import sce_loss
from mae.utils import create_norm
from torch_geometric.utils import dropout_edge
from torch_geometric.utils import add_self_loops, to_dense_batch


def find_continuous_lengths(lst):
    result = []
    current_length = 1

    for i in range(1, len(lst)):
        if lst[i] == lst[i - 1]:
            current_length = current_length + 1
        else:
            result.append(current_length)
            current_length = 1

    result.append(current_length)

    return result


def partition_list(lengths, lst):
    result = []

    start_index = 0
    for length in lengths:
        end_index = start_index + length
        result.append(lst[start_index:end_index])
        start_index = end_index

    return result


def setup_module(m_type, enc_dec, in_dim, num_hidden, out_dim, num_layers, dropout, activation, residual, norm, nhead, nhead_out, attn_drop, negative_slope=0.2, concat_out=True) -> nn.Module:
    if m_type == "gat":
        mod = GAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "gin":
        mod = GIN(
            in_dim=int(in_dim),
            num_hidden=int(num_hidden),
            out_dim=int(out_dim),
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            residual=residual,
            norm=norm,
            encoding=(enc_dec == "encoding"),
        )
    else:
        raise NotImplementedError

    return mod


class PreModel(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_hidden: int,
        num_layers: int,
        nhead: int,
        nhead_out: int,
        activation: str,
        feat_drop: float,
        attn_drop: float,
        negative_slope: float,
        residual: bool,
        norm: Optional[str],
        mask_rate: float = 0.3,
        encoder_type: str = "gat",
        decoder_type: str = "gat",
        loss_fn: str = "sce",
        drop_edge_rate: float = 0.0,
        replace_rate: float = 0.1,
        alpha_l: float = 2,
        concat_hidden: bool = False,
        trans_num_layers: int = 1,
        dec_num_layers: int = 1,
        pe_dim: int = 24,
        enc_type: str = "normal",  # normal parallel summing
        dec_type: str = "gnn",  # gnn mlp gt ct
    ):
        super(PreModel, self).__init__()
        self._mask_rate = mask_rate
        self._encoder_type = encoder_type
        self._decoder_type = decoder_type
        self._drop_edge_rate = drop_edge_rate
        self._output_hidden_size = num_hidden
        self._concat_hidden = concat_hidden

        self.enc_type = enc_type
        self.dec_type = dec_type

        self._replace_rate = replace_rate
        self._mask_token_rate = 1 - self._replace_rate

        assert num_hidden % nhead == 0
        assert num_hidden % nhead_out == 0
        if encoder_type in ("gat", "dotgat"):
            enc_num_hidden = num_hidden // nhead
            enc_nhead = nhead
        else:
            enc_num_hidden = num_hidden
            enc_nhead = 1

        dec_in_dim = num_hidden
        dec_num_hidden = num_hidden // nhead_out if decoder_type in (
            "gat", "dotgat") else num_hidden

        # build gnn_encoder
        self.gnn_encoder = setup_module(
            m_type=encoder_type,
            enc_dec="encoding",
            in_dim=in_dim,
            num_hidden=enc_num_hidden,
            out_dim=enc_num_hidden,
            num_layers=num_layers,
            nhead=enc_nhead,
            nhead_out=enc_nhead,
            concat_out=True,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
        )

        # build decoder for attribute prediction
        self.gnn_decoder = setup_module(
            m_type=decoder_type,
            enc_dec="decoding",
            in_dim=dec_in_dim,
            num_hidden=dec_num_hidden,
            out_dim=in_dim,
            num_layers=dec_num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
            concat_out=True,
        )

        self.mlp_decoder = nn.Sequential(
            nn.Linear(dec_in_dim, dec_num_hidden),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(dec_num_hidden, in_dim)
        )
        self.liner_decoder = nn.Linear(dec_in_dim, in_dim)
        self.encoder_liner = nn.Linear(in_dim, dec_in_dim)

        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))
        if concat_hidden:
            self.encoder_to_decoder = nn.Linear(
                dec_in_dim * num_layers, dec_in_dim, bias=False)
        else:
            self.encoder_to_decoder = nn.Linear(
                dec_in_dim, dec_in_dim, bias=False)

        self.gt_layer = GtLayer(
            in_dim=dec_in_dim, trans_num_layers=trans_num_layers)
        self.gt_decoder_layer = GtDecoderLayer(
            in_dim=dec_in_dim, dec_num_layers=dec_num_layers)
        self.ct_decoder_layer = CtDecoderLayer(
            in_dim=dec_in_dim, dec_num_layers=dec_num_layers)

        self.mixed_layer = MixedLayer(
            dec_in_dim, residual, norm, pe_dim, num_layers=num_layers)
        self.group_layer = GroupLayer(
            dec_in_dim, residual, norm, pe_dim, num_layers=num_layers)

        self.node_liner = nn.Linear(dec_in_dim, dec_in_dim-pe_dim, bias=False)
        self.dec_node_liner = nn.Linear(
            dec_in_dim, dec_in_dim-pe_dim, bias=False)

        # * setup loss function
        self.criterion = self.setup_loss_fn(loss_fn, alpha_l)

    @property
    def output_hidden_dim(self):
        return self._output_hidden_size

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion

    def encoding_mask_noise(self, x, mask_rate=0.3):
        num_nodes = x.shape[0]
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)

        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes, _ = perm[: num_mask_nodes].sort()
        keep_nodes, _ = perm[num_mask_nodes:].sort()

        if self._replace_rate > 0:
            num_noise_nodes = int(self._replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(
                self._mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(
                self._replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[
                :num_noise_nodes]

            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token

        return out_x, (mask_nodes, keep_nodes)

    def get_mask_subgraph(self, list, mask_rate, epoch, max_epoch, subgraph_lambda=2):
        batch_list, idx_list = list
        continuous_list = find_continuous_lengths(batch_list)
        idx_parts = partition_list(continuous_list, idx_list)
        num_subgraph = len(continuous_list)
        # -----------
        num_mask_subgraph = int(mask_rate*num_subgraph)
        mask = np.hstack([
            np.zeros(num_subgraph - num_mask_subgraph),
            np.ones(num_mask_subgraph),
        ])
        np.random.shuffle(mask)
        mask = torch.tensor(mask).bool()
        # -----------
        num_sub_nodes = torch.tensor(continuous_list).int()
        num_mask1 = (num_sub_nodes*mask_rate).int()

        shuffle_mask_sub = torch.zeros(num_subgraph).int()
        shuffle_mask_sub[mask] = num_sub_nodes[mask]
        num_mask2 = shuffle_mask_sub
        lam = (epoch/max_epoch)**subgraph_lambda
        num_mask = torch.round(lam*num_mask1 + (1-lam)*num_mask2).int()

        mask_nodes_list = []
        for i in range(len(idx_parts)):
            mask_nodes_list = mask_nodes_list + \
                idx_parts[i][:num_mask[i].item()]

        total_perm = torch.arange(len(batch_list)).int()
        mask_nodes, _ = torch.tensor(mask_nodes_list).int().sort()
        perm_mask = torch.isin(total_perm, mask_nodes)
        keep_nodes = total_perm[~perm_mask]

        return mask_nodes, keep_nodes

    def forward(self, batch_g, list, epoch, max_epoch):
        # ---- attribute reconstruction ----
        loss, node_emb = self.mask_attr_prediction(
            batch_g, list, epoch, max_epoch)
        loss_item = {"loss": loss.item()}
        return loss, loss_item, node_emb

    def mask_attr_prediction(self, batch_g, list, epoch, max_epoch):
        x, edge_index = batch_g.x, batch_g.edge_index

        batch_list, _ = list
        if len(batch_list) > 0:
            mask_nodes, keep_nodes = self.get_mask_subgraph(
                list, self._mask_rate, epoch, max_epoch)
            use_x = x.clone()
            use_x[mask_nodes] = 0.0
            use_x = use_x + self.enc_mask_token
        else:
            use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(
                x, self._mask_rate)

        if self._drop_edge_rate > 0:
            use_edge_index, masked_edges = dropout_edge(
                edge_index, self._drop_edge_rate)
            use_edge_index = add_self_loops(use_edge_index)[0]
        else:
            use_edge_index = edge_index

        gnn_enc_rep, all_hidden = self.gnn_encoder(
            use_x, use_edge_index, return_hidden=True)
        if self._concat_hidden:
            gnn_enc_rep = torch.cat(all_hidden, dim=1)

        if self.enc_type == "parallel":
            enc_rep = gnn_enc_rep.clone()

            keep_x = self.encoder_liner(use_x[keep_nodes])
            batch = batch_g.batch
            pe = batch_g.pe
            keep_batch = batch[keep_nodes]
            keep_pe = pe[keep_nodes]
            t_rep = torch.cat((self.node_liner(keep_x), keep_pe), dim=-1)
            h, h_mask = to_dense_batch(t_rep, keep_batch)

            pad_enc = self.gt_layer(h)
            gt_out = pad_enc[h_mask]
            enc_rep[keep_nodes] = enc_rep[keep_nodes] + gt_out
        elif self.enc_type == "mixed":
            enc_x = self.encoder_liner(use_x)
            batch = batch_g.batch
            pe = batch_g.pe
            keep_batch = batch[keep_nodes]
            keep_pe = pe[keep_nodes]
            mix_out = self.mixed_layer(
                enc_x, use_edge_index, keep_batch, keep_pe, keep_nodes)
            enc_rep = mix_out
        elif self.enc_type == "group":
            enc_x = self.encoder_liner(use_x)
            batch = batch_g.batch
            pe = batch_g.pe
            keep_batch = batch[keep_nodes]
            keep_pe = pe[keep_nodes]
            group_out = self.group_layer(
                enc_x, use_edge_index, keep_batch, keep_pe, keep_nodes)
            enc_rep = group_out
        else:
            enc_rep = gnn_enc_rep.clone()

            batch = batch_g.batch
            pe = batch_g.pe
            keep_rep = enc_rep[keep_nodes]
            keep_batch = batch[keep_nodes]
            keep_pe = pe[keep_nodes]
            t_rep = torch.cat((self.node_liner(keep_rep), keep_pe), dim=-1)
            h, h_mask = to_dense_batch(t_rep, keep_batch)

            pad_enc = self.gt_layer(h)
            gt_out = pad_enc[h_mask]
            enc_rep[keep_nodes] = gt_out

        node_emb = enc_rep.clone()
        # ---- attribute reconstruction ----
        rep = self.encoder_to_decoder(enc_rep)

        if self.dec_type not in ("mlp"):
            # * remask, re-mask
            rep[mask_nodes] = 0

        if self.dec_type == "gnn":
            recon = self.gnn_decoder(rep, use_edge_index)
        elif self.dec_type == "gt":
            t_recon = torch.cat((self.dec_node_liner(rep), pe), dim=-1)
            d, d_mask = to_dense_batch(t_recon, batch_g.batch)
            trans_d = self.gt_decoder_layer(d)
            dec = trans_d[d_mask]
            recon = self.liner_decoder(dec)
        elif self.dec_type == "ct":

            t_recon = torch.cat((self.dec_node_liner(rep), pe), dim=-1)

            mask_recon = t_recon[mask_nodes]
            mask_batch = batch_g.batch[mask_nodes]
            isin_mask = torch.isin(batch_g.batch, mask_batch)
            sub_batch = batch_g.batch[isin_mask]
            sub_recon = t_recon[isin_mask]

            d, _ = to_dense_batch(sub_recon, sub_batch)
            m, m_mask = to_dense_batch(mask_recon, mask_batch)
            trans_m = self.ct_decoder_layer(d, m)

            dec = trans_m[m_mask]
            m_rep = rep.clone()
            m_rep[mask_nodes] = dec
            recon = self.liner_decoder(m_rep)
        else:
            recon = self.mlp_decoder(rep)

        x_init = x[mask_nodes]
        x_rec = recon[mask_nodes]

        loss = self.criterion(x_rec, x_init)
        return loss, node_emb

    def embed(self, batch_g):
        x, edge_index = batch_g.x, batch_g.edge_index
        rep = self.gnn_encoder(x, edge_index)

        if self.enc_type == "parallel":
            keep_x = self.encoder_liner(x)
            batch = batch_g.batch
            pe = batch_g.pe
            enc_rep = torch.cat((self.node_liner(keep_x), pe), dim=-1)

            h_pad, h_mask = to_dense_batch(enc_rep, batch)
            gt_out = self.gt_layer(h_pad)
            out = gt_out[h_mask] + rep
        elif self.enc_type == "mixed":
            keep_x = self.encoder_liner(x)
            batch = batch_g.batch
            pe = batch_g.pe
            out = self.mixed_layer(keep_x, edge_index, batch, pe)
        elif self.enc_type == "group":
            keep_x = self.encoder_liner(x)
            batch = batch_g.batch
            pe = batch_g.pe
            out = self.group_layer(keep_x, edge_index, batch, pe)
        else:
            batch = batch_g.batch
            pe = batch_g.pe
            enc_rep = torch.cat((self.node_liner(rep), pe), dim=-1)
            h_pad, h_mask = to_dense_batch(enc_rep, batch)
            gt_out = self.gt_layer(h_pad)
            out = gt_out[h_mask]
        return out
