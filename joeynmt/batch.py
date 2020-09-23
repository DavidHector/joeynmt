# coding: utf-8
import torch
import torch.nn as nn
"""
Implementation of a mini-batch.
"""

class BatchBatch:
    """Object for conforming to Batch requirements (batch.src must be defined an return src and src_lengths,
    as well as batch.trg)"""
    # TODO: pad src, so that stack has all same dimensions. use mask to restrict training to correct data
    def __init__(self, batch, pad_index, use_cuda=False):
        self.src, self.trg = torch.stack([x[0] for x in batch]), [x[1] for x in batch]
        self.src_mask = (self.src != pad_index).unsqueeze(1)
        self.nseqs = None  # self.src.size(0)
        self.trg_input = None
        self.trg = None
        self.trg_mask = None
        self.trg_lengths = None
        self.ntokens = None
        self.use_cuda = use_cuda

        if hasattr(batch, "trg"):
            trg, trg_lengths = batch.trg
            # trg_input is used for teacher forcing, last one is cut off
            self.trg_input = trg[:, :-1]
            self.trg_lengths = trg_lengths
            # trg is used for loss computation, shifted by one since BOS
            self.trg = trg[:, 1:]
            # we exclude the padded areas from the loss computation
            self.trg_mask = (self.trg_input != pad_index).unsqueeze(1)
            self.ntokens = (self.trg != pad_index).data.sum().item()

        if use_cuda:
            self._make_cuda()


class Batch:
    """Object for holding a batch of data with mask during training.
    Input is a batch from a torch text iterator.
    """

    def __init__(self, torch_batch, pad_index, use_cuda=False):
        """
        Create a new joey batch from a torch batch.
        This batch extends torch text's batch attributes with src and trg
        length, masks, number of non-padded tokens in trg.
        Furthermore, it can be sorted by src length.

        :param torch_batch:
        :param pad_index:
        :param use_cuda:
        """
        batchsrctemp = []
        for i in torch_batch.src:
            batchsrctemp.append(i.transpose(0, 1))
        src_temp = nn.utils.rnn.pad_sequence(batchsrctemp, batch_first=True, padding_value=pad_index) # list10 x tensor201 x len
        #self.src, self.src_lengths = torch_batch.src
        self.src = src_temp # .transpose(1, 2) # was ist das? Src.length sollte 201 sein (=embed) 201x1x?x1
        self.src_lengths = torch.zeros(self.src.shape[0])
        self.src_mask = torch.full((self.src.shape[0], self.src.shape[1]), True, dtype=bool)
        self.pad_vector = torch.full((self.src.shape[2],), pad_index, dtype=self.src.dtype)
        for i in range(len(self.src)):
            self.src_lengths[i] = len(self.src[i])
            for j in range(len(self.src[i])):
                if (self.src[i][j] == self.pad_vector).all():
                    self.src_mask[i][j] = False
        self.src_mask = self.src_mask.unsqueeze(1)
        # Was wir eigentlich bräuchten, ist eine Maske 10xL, die für die PaddingVEKTOREN einen Eintrag False hat
        # self.src_mask = (self.src.transpose(1, 2) != pad_index).unsqueeze(1)  # mask for attention sollte 10x1xL sein, nein ,weil wir keine indices haben, sondern die encodings
        self.nseqs = self.src.size(0)
        self.trg_input = None
        self.trg = None
        self.trg_mask = None
        self.trg_lengths = None
        self.ntokens = None
        self.use_cuda = use_cuda

        if hasattr(torch_batch, "trg"):
            trg, trg_lengths = torch_batch.trg
            # trg_input is used for teacher forcing, last one is cut off
            self.trg_input = trg[:, :-1]
            self.trg_lengths = trg_lengths
            # trg is used for loss computation, shifted by one since BOS
            self.trg = trg[:, 1:]
            # we exclude the padded areas from the loss computation
            self.trg_mask = (self.trg_input != pad_index).unsqueeze(1)
            self.ntokens = (self.trg != pad_index).data.sum().item()

        if use_cuda:
            self._make_cuda()

    def _make_cuda(self):
        """
        Move the batch to GPU

        :return:
        """
        self.src = self.src.cuda()
        self.src_mask = self.src_mask.cuda()

        if self.trg_input is not None:
            self.trg_input = self.trg_input.cuda()
            self.trg = self.trg.cuda()
            self.trg_mask = self.trg_mask.cuda()

    def sort_by_src_lengths(self):
        """
        Sort by src length (descending) and return index to revert sort

        :return:
        """
        _, perm_index = self.src_lengths.sort(0, descending=True)
        rev_index = [0]*perm_index.size(0)
        for new_pos, old_pos in enumerate(perm_index.cpu().numpy()):
            rev_index[old_pos] = new_pos

        sorted_src_lengths = self.src_lengths[perm_index]
        sorted_src = self.src[perm_index]
        sorted_src_mask = self.src_mask[perm_index]
        if self.trg_input is not None:
            sorted_trg_input = self.trg_input[perm_index]
            sorted_trg_lengths = self.trg_lengths[perm_index]
            sorted_trg_mask = self.trg_mask[perm_index]
            sorted_trg = self.trg[perm_index]

        self.src = sorted_src
        self.src_lengths = sorted_src_lengths
        self.src_mask = sorted_src_mask

        if self.trg_input is not None:
            self.trg_input = sorted_trg_input
            self.trg_mask = sorted_trg_mask
            self.trg_lengths = sorted_trg_lengths
            self.trg = sorted_trg

        if self.use_cuda:
            self._make_cuda()

        return rev_index
