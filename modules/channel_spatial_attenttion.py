# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# https://github.com/facebookresearch/brainmagick/blob/main/bm/models/common.py
# Paper: Decoding speech from non-invasive recordings of brain activity
import mne
import torch
from torch import nn
import logging
import math
import typing as tp
logger = logging.getLogger(__name__)


class PositionGetter:
    INVALID = -0.1

    def __init__(self) -> None:
        self._cache: tp.Dict[int, torch.Tensor] = {}
        self._invalid_names: tp.Set[str] = set()

    def get_recording_layout(self, recording):
        index = recording.recording_index
        if index in self._cache:
            return self._cache[index]
        else:
            info = recording.mne_info
            layout = mne.find_layout(info)
            indexes: tp.List[int] = []
            valid_indexes: tp.List[int] = []
            for meg_index, name in enumerate(info.ch_names):
                name = name.rsplit("-", 1)[0]
                try:
                    indexes.append(layout.names.index(name))
                except ValueError:
                    if name not in self._invalid_names:
                        logger.warning(
                            "Channels %s not in layout for recording %s of %s.",
                            name,
                            recording.study_name(),
                            recording.recording_uid)
                        self._invalid_names.add(name)
                else:
                    valid_indexes.append(meg_index)

            positions = torch.full((len(info.ch_names), 2), self.INVALID)
            x, y = layout.pos[indexes, :2].T
            x = (x - x.min()) / (x.max() - x.min())
            y = (y - y.min()) / (y.max() - y.min())
            x = torch.from_numpy(x).float()
            y = torch.from_numpy(y).float()
            positions[valid_indexes, 0] = x
            positions[valid_indexes, 1] = y
            self._cache[index] = positions
            return positions

    def get_positions(self, batch):
        meg = batch.meg
        B, C, F, T = meg.shape
        positions = torch.full((B, C, 2), self.INVALID, device=meg.device)
        for idx in range(1):
        # for idx in range(batch):
            recording = batch._recordings[idx]
            rec_pos = self.get_recording_layout(recording)
            positions[:, :, :len(rec_pos)] = rec_pos.to(meg.device)
        return positions

    def is_invalid(self, positions):
        return (positions == self.INVALID).all(dim=-1)


class FourierEmb(nn.Module):
    """
    Fourier positional embedding.
    Unlike trad. embedding this is not using exponential periods
    for cosines and sinuses, but typical `2 pi k` which can represent
    any function over [0, 1]. As this function would be necessarily periodic,
    we take a bit of margin and do over [-0.2, 1.2].
    """
    def __init__(self, dimension: int = 128, margin: float = 0.2):
        super().__init__()
        n_freqs = (dimension // 2)**0.5
        assert int(n_freqs ** 2 * 2) == dimension
        self.dimension = dimension
        self.margin = margin

    def forward(self, positions):
        *O, D = positions.shape
        assert D == 2
        *O, D = positions.shape
        n_freqs = (self.dimension // 2)**0.5
        freqs_y = torch.arange(n_freqs).to(positions)
        freqs_x = freqs_y[:, None]
        width = 1 + 2 * self.margin
        positions = positions + self.margin
        p_x = 2 * math.pi * freqs_x / width
        p_y = 2 * math.pi * freqs_y / width
        positions = positions[..., None, None, :]
        loc = (positions[..., 0] * p_x + positions[..., 1] * p_y).view(*O, -1)
        emb = torch.cat([
            torch.cos(loc),
            torch.sin(loc),
        ], dim=-1)
        return emb


class ChannelDropout(nn.Module):
    def __init__(self, dropout: float = 0.1, rescale: bool = True):
        """
        Args:
            dropout: dropout radius in normalized [0, 1] coordinates.
            rescale: at valid, rescale all channels.
        """
        super().__init__()
        self.dropout = dropout
        self.rescale = rescale
        self.position_getter = PositionGetter()

    def forward(self, meg, batch):
        if not self.dropout:
            return meg

        B, C, F, T = meg.shape
        meg = meg.clone()
        positions = self.position_getter.get_positions(batch)
        valid = (~self.position_getter.is_invalid(positions)).float()
        meg = meg * valid[:, :, None, None]

        if self.training:
            center_to_ban = torch.rand(2, device=meg.device)
            kept = (positions - center_to_ban).norm(dim=-1) > self.dropout
            meg = meg * kept.float()[:, :, None, None]
            if self.rescale:
                proba_kept = torch.zeros(B, C, device=meg.device)
                n_tests = 100
                for _ in range(n_tests):
                    center_to_ban = torch.rand(2, device=meg.device)
                    kept = (positions - center_to_ban).norm(dim=-1) > self.dropout
                    proba_kept += kept.float() / n_tests
                meg = meg / (1e-8 + proba_kept[:, :, None, None])

        return meg


class ChannelMerger(nn.Module):
    def __init__(self, chout: int, pos_dim: int = 128,
                 dropout: float = 0.2, usage_penalty: float = 0.,
                 n_subjects: int = 200, per_subject: bool = False):
        super().__init__()
        assert pos_dim % 4 == 0
        self.position_getter = PositionGetter()
        self.per_subject = per_subject
        if self.per_subject:
            self.heads = nn.Parameter(torch.randn(n_subjects, chout, pos_dim, requires_grad=True))
        else:
            self.heads = nn.Parameter(torch.randn(chout, pos_dim, requires_grad=True))
        self.heads.data /= pos_dim ** 0.5
        self.dropout = dropout
        self.embedding = FourierEmb(pos_dim)
        self.usage_penalty = usage_penalty
        self._penalty = torch.tensor(0.)

    @property
    def training_penalty(self):
        return self._penalty.to(next(self.parameters()).device)




    def forward(self, meg, batch):
        B, C, F, T = meg.shape
        meg = meg.clone()
        positions = self.position_getter.get_positions(batch)
        embedding = self.embedding(positions)
        score_offset = torch.zeros(B, C, device=meg.device)
        score_offset[self.position_getter.is_invalid(positions)] = float('-inf')

        if self.training and self.dropout:
            center_to_ban = torch.rand(2, device=meg.device)
            radius_to_ban = self.dropout
            banned = (positions - center_to_ban).norm(dim=-1) <= radius_to_ban
            score_offset[banned] = float('-inf')

        if self.per_subject:
            _, cout, pos_dim = self.heads.shape
            subject = batch.subject_index
            heads = self.heads.gather(0, subject.view(-1, 1, 1).expand(-1, cout, pos_dim))
        else:
            heads = self.heads[None].expand(B, -1, -1)

        scores = torch.einsum("bcd,bod->boc", embedding, heads)
        scores += score_offset[:, None]
        weights = torch.softmax(scores, dim=2)
        out_weights = weights[0,:,:].clone().cpu().detach().numpy()
        out = torch.einsum("bcft,boc->boft", meg, weights)

        if self.training and self.usage_penalty > 0.:
            usage = weights.mean(dim=(0, 1)).sum()
            self._penalty = self.usage_penalty * usage


        return out, out_weights

