# coding: utf-8
"""
Data module
"""
import sys
import random
import os
import os.path
from typing import Optional
import torch
from torchtext.datasets import TranslationDataset
from torchtext import data
from torchtext.data import Dataset, Iterator, Field
import torchaudio # new

from joeynmt.constants import UNK_TOKEN, EOS_TOKEN, BOS_TOKEN, PAD_TOKEN
from joeynmt.vocabulary import build_vocab, Vocabulary
from torchtext.data import get_tokenizer
from torch.utils.data import DataLoader
from .datasets.commonvoice import COMMONVOICE


class Noprocessfield(Field):
    def process(self, batch, device):
        return batch


def preprocess_data_single_entry(input_tuple, type="train", letter_width=0.02, hop_length=400):
    # letter_width=length of single spoken letter in s
    # hop_length = frequency resolution of spectrogram (also determines the letter width)

    input_audio, input_samplerate, input_dict = input_tuple[0], input_tuple[1], input_tuple[2]

    letter_sample_rate = int(hop_length/letter_width)
    downsampler = torchaudio.transforms.Resample(input_samplerate, letter_sample_rate, resampling_method='sinc_interpolation')
    input_audio = downsampler(input_audio)
    spectrogram = torchaudio.transforms.Spectrogram(n_fft=hop_length, hop_length=hop_length)(input_audio)
    sentence = input_dict['sentence']

    tokenizer = get_tokenizer("basic_english")
    tokens = tokenizer(sentence)
    sentence = " ".join(tokens)

    return spectrogram, sentence


def preprocess_data(input_list, type="train", letter_width=0.02, hop_length=400):
    data = []
    for i in input_list:
        data.append(preprocess_data_single_entry(i, type=type, letter_width=letter_width, hop_length=hop_length))
    return data


def reformat_data(data, data_torchaudio, trg_min_freq, trg_max_size, trg_vocab_file, lowercase=True):
    train_iter = data  # make_data_iter(train_data,
    #   batch_size=self.batch_size,
    #  batch_type=self.batch_type,
    # train=True, shuffle=self.shuffle)
    tok_fun = lambda s: s.split()

    src_field = Noprocessfield(sequential=False, use_vocab=False, dtype=torch.double, include_lengths=True)
    trg_field = Field(init_token=BOS_TOKEN, eos_token=EOS_TOKEN,
                      pad_token=PAD_TOKEN, tokenize=tok_fun,
                      unk_token=UNK_TOKEN,
                      batch_first=True, lower=lowercase,
                      include_lengths=True)
    trg_vocab = build_vocab(min_freq=trg_min_freq, max_size=trg_max_size, dataset=data_torchaudio, vocab_file=trg_vocab_file)
    trg_field.vocab = trg_vocab

    entry_list = []
    for i, batch in enumerate(iter(train_iter)):
        # reactivate training
        entry_list.append(Entry(batch[0][0].squeeze(), batch[0][1]))
    train_data = Dataset(entry_list, [('src', src_field), ('trg', trg_field)])
    return train_data, trg_vocab, src_field, trg_field


def load_data(data_cfg: dict) -> (Dataset, Dataset, Optional[Dataset],
                                  Vocabulary, Vocabulary):
    """
    Load train, dev and optionally test data as specified in configuration.
    Vocabularies are created from the training set with a limit of `voc_limit`
    tokens and a minimum token frequency of `voc_min_freq`
    (specified in the configuration dictionary).

    The training data is filtered to include sentences up to `max_sent_length`
    on source and target side.

    If you set ``random_train_subset``, a random selection of this size is used
    from the training set instead of the full training set.

    :param data_cfg: configuration dictionary for data
        ("data" part of configuation file)
    :return:
        - train_data: training dataset
        - dev_data: development dataset
        - test_data: testdata set if given, otherwise None
        - src_vocab: source vocabulary extracted from training data
        - trg_vocab: target vocabulary extracted from training data
    """
    # load data from files
    src_lang = data_cfg["src"]
    trg_lang = data_cfg["trg"]
    train_path = data_cfg["train"]
    dev_path = data_cfg["dev"]
    test_path = data_cfg.get("test", None)
    level = data_cfg["level"]
    lowercase = data_cfg["lowercase"]
    max_sent_length = data_cfg["max_sent_length"]
    tok_fun = lambda s: list(s) if level == "char" else s.split()

    trg_max_size = data_cfg.get("trg_voc_limit", sys.maxsize)
    trg_min_freq = data_cfg.get("trg_voc_min_freq", 1)
    trg_vocab_file = data_cfg.get("trg_vocab", None)
    language = data_cfg.get("language", "esperanto")

    #train_data_torchaudio = torchaudio.datasets.COMMONVOICE(train_path, url=language, download=True, tsv='minitrain.tsv')
    train_data_torchaudio = COMMONVOICE('CommonVoice', language=language, download=True, tsv=train_path)
    # changed the dataset from a Translation Dataset to torchaudio dataset.
    # created DataLoader which can be used in existing data training loop
    # made preprocessing function
    # Done: make vocabulary manually, using Vocabulary class (only for target)
    train_data = DataLoader(train_data_torchaudio, batch_size=1, shuffle=False, collate_fn= lambda x: preprocess_data(x, type="train"))

    # random_train_subset = data_cfg.get("random_train_subset", -1)
    # # Todo: delete this bc unnecessary? + split doesnt work with our train_data object
    # if random_train_subset > -1:
    #     # select this many training examples randomly and discard the rest
    #     keep_ratio = random_train_subset / len(train_data)
    #     keep, _ = train_data.split(
    #         split_ratio=[keep_ratio, 1 - keep_ratio],
    #         random_state=random.getstate())
    #     train_data = keep

    dev_data_torchaudio = COMMONVOICE('CommonVoice', language=language, tsv=dev_path)#, download=True)
    dev_data = DataLoader(dev_data_torchaudio, batch_size=1, shuffle=False, collate_fn= lambda x: preprocess_data(x, type="dev"))

    test_data = None
    if test_path is not None:
        # check if target exists
        test_data_torchaudio = COMMONVOICE('CommonVoice', language=language,tsv=test_path)#, download=True)
        test_data = DataLoader(test_data_torchaudio, batch_size=1, shuffle=False, collate_fn=lambda x: preprocess_data(x, type="test"))
        test_data, test_trg_vocab, test_src_field, test_trg_field = reformat_data(test_data, test_data_torchaudio,
                                                                              trg_min_freq, trg_max_size,
                                                                              trg_vocab_file, lowercase=lowercase)

    # Todo: same processing we did for train_data, has to be used for dev_data and valid_data
    train_data, trg_vocab, src_field, trg_field = reformat_data(train_data, train_data_torchaudio, trg_min_freq, trg_max_size, trg_vocab_file, lowercase=lowercase)
    dev_data, dev_trg_vocab, dev_src_field, dev_trg_field = reformat_data(dev_data, dev_data_torchaudio, trg_min_freq, trg_max_size, trg_vocab_file, lowercase=lowercase)

    return train_data, dev_data, test_data, trg_vocab


# pylint: disable=global-at-module-level
global max_src_in_batch, max_tgt_in_batch


# pylint: disable=unused-argument,global-variable-undefined
def token_batch_size_fn(new, count, sofar):
    """Compute batch size based on number of tokens (+padding)."""
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    src_elements = count * max_src_in_batch
    if hasattr(new, 'trg'):  # for monolingual data sets ("translate" mode)
        max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
        tgt_elements = count * max_tgt_in_batch
    else:
        tgt_elements = 0
    return max(src_elements, tgt_elements)


def make_data_iter(dataset: Dataset,
                   batch_size: int,
                   batch_type: str = "sentence",
                   train: bool = False,
                   shuffle: bool = False) -> Iterator:
    """
    Returns a torchtext iterator for a torchtext dataset.

    :param dataset: torchtext dataset containing src and optionally trg
    :param batch_size: size of the batches the iterator prepares
    :param batch_type: measure batch size by sentence count or by token count
    :param train: whether it's training time, when turned off,
        bucketing, sorting within batches and shuffling is disabled
    :param shuffle: whether to shuffle the data before each epoch
        (no effect if set to True for testing)
    :return: torchtext iterator
    """

    batch_size_fn = token_batch_size_fn if batch_type == "token" else None

    if train:
        # optionally shuffle and sort during training
        data_iter = data.BucketIterator(
            repeat=False, sort=False, dataset=dataset,
            batch_size=batch_size, batch_size_fn=batch_size_fn,
            train=True, sort_within_batch=True,
            sort_key=lambda x: len(x.src), shuffle=shuffle)
    else:
        # don't sort/shuffle for validation/inference
        data_iter = data.BucketIterator(
            repeat=False, dataset=dataset,
            batch_size=batch_size, batch_size_fn=batch_size_fn,
            sort_key=lambda x: len(x.src),
            train=False, sort=False)

    return data_iter

# Todo: Neue Audiotextdaten-Klasse schreiben, damit wir nicht alle Abhängigkeiten ändern müssen


class Entry():
    def __init__(self, src, trg):
        self.src = src
        self.trg = trg


class MonoDataset(Dataset):
    """Defines a dataset for machine translation without targets."""

    @staticmethod
    def sort_key(ex):
        return len(ex.src)

    def __init__(self, path: str, ext: str, field: Field, **kwargs) -> None:
        """
        Create a monolingual dataset (=only sources) given path and field.

        :param path: Prefix of path to the data file
        :param ext: Containing the extension to path for this language.
        :param field: Containing the fields that will be used for data.
        :param kwargs: Passed to the constructor of data.Dataset.
        """

        fields = [('src', field)]

        if hasattr(path, "readline"):  # special usage: stdin
            src_file = path
        else:
            src_path = os.path.expanduser(path + ext)
            src_file = open(src_path)

        examples = []
        for src_line in src_file:
            src_line = src_line.strip()
            if src_line != '':
                examples.append(data.Example.fromlist(
                    [src_line], fields))

        src_file.close()

        super(MonoDataset, self).__init__(examples, fields, **kwargs)
