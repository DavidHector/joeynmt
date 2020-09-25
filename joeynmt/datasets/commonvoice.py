"""
Forked from https://github.com/pytorch/audio/blob/master/torchaudio/datasets/commonvoice.py
to fix the tsv target path and update the dataset version and URL.
"""
import os
import pathlib
from typing import List, Dict, Tuple

import torchaudio
from torchaudio.datasets.utils import download_url, extract_archive, unicode_csv_reader
from torch import Tensor
from torch.utils.data import Dataset

# Default TSV should be one of
# dev.tsv
# invalidated.tsv
# other.tsv
# test.tsv
# train.tsv
# validated.tsv

LANGUAGE = "english"
VERSION = "cv-corpus-5.1-2020-06-22"
TSV = "train.tsv"
_CHECKSUMS = {
    "https://cdn.commonvoice.mozilla.org/cv-corpus-5.1-2020-06-22/pa-IN.tar.gz":
    None,
}


def load_commonvoice_item(line: List[str],
                          header: List[str],
                          path: str,
                          folder_audio: str) -> Tuple[Tensor, int, Dict[str, str]]:
    # Each line as the following data:
    # client_id, path, sentence, up_votes, down_votes, age, gender, accent

    assert header[1] == "path"
    fileid = line[1]

    filename = os.path.join(path, folder_audio, fileid)

    waveform, sample_rate = torchaudio.load(filename)

    dic = dict(zip(header, line))

    return waveform, sample_rate, dic


class COMMONVOICE(Dataset):
    """
    Create a Dataset for CommonVoice. Each item is a tuple of the form:
    (waveform, sample_rate, dictionary)
    where dictionary is a dictionary built from the tsv file with the following keys:
    client_id, path, sentence, up_votes, down_votes, age, gender, accent.
    """

    _ext_txt = ".txt"
    _ext_audio = ".mp3"
    _folder_audio = "clips"

    def __init__(self,
                 root: str,
                 tsv: str = TSV,
                 language: str = LANGUAGE,
                 version: str = VERSION,
                 download: bool = False) -> None:

        languages = {
            "tatar": "tt",
            "english": "en",
            "german": "de",
            "french": "fr",
            "welsh": "cy",
            "breton": "br",
            "chuvash": "cv",
            "turkish": "tr",
            "kyrgyz": "ky",
            "irish": "ga-IE",
            "kabyle": "kab",
            "catalan": "ca",
            "taiwanese": "zh-TW",
            "slovenian": "sl",
            "italian": "it",
            "dutch": "nl",
            "hakha chin": "cnh",
            "esperanto": "eo",
            "estonian": "et",
            "persian": "fa",
            "portuguese": "pt",
            "basque": "eu",
            "spanish": "es",
            "chinese": "zh-CN",
            "mongolian": "mn",
            "sakha": "sah",
            "dhivehi": "dv",
            "kinyarwanda": "rw",
            "swedish": "sv-SE",
            "russian": "ru",
            "indonesian": "id",
            "arabic": "ar",
            "tamil": "ta",
            "interlingua": "ia",
            "latvian": "lv",
            "japanese": "ja",
            "votic": "vot",
            "abkhaz": "ab",
            "cantonese": "zh-HK",
            "romansh sursilvan": "rm-sursilv"
        }

        language = languages.get(language, language)
        ext_archive = ".tar.gz"
        base_url = "https://cdn.commonvoice.mozilla.org"
        url = os.path.join(base_url, version, language + ext_archive)

        basename = os.path.basename(url)
        archive = os.path.join(root, basename)

        basename = basename.rsplit(".", 2)[0]
        folder_in_archive = os.path.join(version, language)

        self._path = os.path.join(root, folder_in_archive)

        if download:
            pathlib.Path(root).mkdir(parents=True, exist_ok=True)
            if not os.path.isdir(self._path):
                if not os.path.isfile(archive):
                    checksum = _CHECKSUMS.get(url, None)
                    download_url(url, root, hash_value=checksum)
                extract_archive(archive)
                os.remove(archive)

        self._tsv = os.path.join(self._path, tsv)

        with open(self._tsv, "r") as tsv:
            walker = unicode_csv_reader(tsv, delimiter="\t")
            self._header = next(walker)
            self._walker = list(walker)

    def __getitem__(self, n: int) -> Tuple[Tensor, int, Dict[str, str]]:
        line = self._walker[n]
        return load_commonvoice_item(line, self._header, self._path, self._folder_audio)

    def __len__(self) -> int:
        return len(self._walker)
