# &nbsp; ![Joey-S2T](joey-small.png) Joey NMT for Speech2Text

## Goal and Purpose
Joey NMT was originally meant for translation purposes. Our goal was to alleviate experimenting with speech2text, which is why we modified the original Codebase to accept and process audio data as source and text data as target transcriptions. 


Check out the detailed [documentation](https://joeynmt.readthedocs.io) of the original project and their [paper](https://arxiv.org/abs/1907.12484). Our paper can be found in [this repository](./Speech_to_Text_in_JoeyNMT.pdf).

## Contributors
Joey-S2T is developed by [Niklas Korz](https://github.com/niklaskorz) (Heidelberg University), [Yoalli Rezepka García](https://github.com/Yrgarcia) (Heidelberg University) and [David Hector](https://github.com/DavidHector) (Heidelberg University).
Joey NMT is developed by [Jasmijn Bastings](https://github.com/bastings) (University of Amsterdam) and [Julia Kreutzer](http://www.cl.uni-heidelberg.de/~kreutzer/) (Heidelberg University).


## Features
- Speech2Text for all languages provided by the [Commonvoice Dataset](https://commonvoice.mozilla.org/en)


## Installation
Joey S2T is built on [PyTorch](https://pytorch.org/), [torchaudio](https://pytorch.org/audio/) and [torchtext](https://github.com/pytorch/text) for Python >= 3.5.

A. From source
  1. Clone this repository:
  `git clone https://github.com/DavidHector/joeynmt.git`
  2. Install joeynmt and it's requirements:
  `cd joeynmt`
  `pip3 install .` (you might want to add `--user` for a local installation).

**Warning!** When running on *GPU* you need to manually install the suitable PyTorch version for your [CUDA](https://developer.nvidia.com/cuda-zone) version. This is described in the [PyTorch installation instructions](https://pytorch.org/get-started/locally/).


### Configuration
Experiments are specified in configuration files, in simple [YAML](http://yaml.org/) format. You can find an example in the `configs` directory.
Edit the language parameter in `stt.yaml` to run Speech2Text for different languages.

Most importantly, the configuration contains the description of the model architecture (e.g. number of hidden units in the encoder RNN) and the training hyperparameters (learning rate, validation frequency etc.).

### Training

#### Start
For training, we recommend running our extension on Google Colab.
Run:

`python3 -m joeynmt train configs/stt.yaml`. 

This will train a model on the training data specified in the config (here: `stt.yaml`), 
validate on validation data, 
and store model parameters, vocabularies, validation outputs and a small number of attention plots in the `model_dir` (also specified in config).

Tip: Be careful not to overwrite models, set `overwrite: False` in the model configuration.

#### Validations
The `validations.txt` file in the model directory reports the validation results at every validation point. 
Models are saved whenever a new best validation score is reached, in `batch_no.ckpt`, where `batch_no` is the number of batches the model has been trained on so far.
`best.ckpt` links to the checkpoint that has so far achieved the best validation score.


#### CPU vs. GPU
For training on a GPU, set `use_cuda` in the config file to `True`. This requires the installation of required CUDA libraries.


### Translating From Audio to Text

#### File Speech2Text
In order to transcribe the contents of an audio file, simply run

`python3 -m joeynmt translate configs/stt.yaml --input_path < my_input.mp3 > `.

The transcriptions will be written to stdout or alternatively`--output_path` if specified.


## Other Projects and Extensions
Here we'll collect projects and repositories that are based on Joey, so you can find inspiration and examples on how to modify and extend the code.

- **Joey Toy Models**. [@bricksdont](https://github.com/bricksdont) built a [collection of scripts](https://github.com/bricksdont/joeynmt-toy-models) showing how to install JoeyNMT, preprocess data, train and evaluate models. This is a great starting point for anyone who wants to run systematic experiments, tends to forget python calls, or doesn't like to run notebook cells! 
- **African NMT**. [@jaderabbit](https://github.com/jaderabbit) started an initiative at the Indaba Deep Learning School 2019 to ["put African NMT on the map"](https://twitter.com/alienelf/status/1168159616167010305). The goal is to build and collect NMT models for low-resource African languages. The [Masakhane repository](https://github.com/masakhane-io/masakhane-mt) contains and explains all the code you need to train JoeyNMT and points to data sources. It also contains benchmark models and configurations that members of Masakhane have built for various African languages. Furthermore, you might be interested in joining the [Masakhane community](https://github.com/masakhane-io/masakhane-community) if you're generally interested in low-resource NLP/NMT.
- **Slack Joey**. [Code](https://github.com/juliakreutzer/slack-joey) to locally deploy a Joey NMT model as chat bot in a Slack workspace. It's a convenient way to probe your model without having to implement an API. And bad translations for chat messages can be very entertaining, too ;)
- **Flask Joey**. [@kevindegila](https://github.com/kevindegila) built a [flask interface to Joey](https://github.com/kevindegila/flask-joey), so you can deploy your trained model in a web app and query it in the browser. 
- **User Study**. We evaluated the code quality of this repository by testing the understanding of novices through quiz questions. Find the details in Section 3 of the [Joey NMT paper](https://arxiv.org/abs/1907.12484).
- **Self-Regulated Interactive Seq2Seq Learning**. Julia Kreutzer and Stefan Riezler. Published at ACL 2019. [Paper](https://arxiv.org/abs/1907.05190) and [Code](https://github.com/juliakreutzer/joeynmt/tree/acl19). This project augments the standard fully-supervised learning regime by weak and self-supervision for a better trade-off of quality and supervision costs in interactive NMT.
- **Speech Joey**. [@Sariyusha](https://github.com/Sariyusha) is giving Joey ears for speech translation. [Code](https://github.com/Sariyusha/speech_joey).
- **Hieroglyph Translation**. Joey NMT was used to translate hieroglyphs in [this IWSLT 2019 paper](https://www.cl.uni-heidelberg.de/statnlpgroup/publications/IWSLT2019.pdf) by Philipp Wiesenbach and Stefan Riezler. They gave Joey NMT multi-tasking abilities. 

## Reference
If you use Joey NMT in a publication or thesis, please cite the following [paper](https://arxiv.org/abs/1907.12484):

```
@inproceedings{kreutzer-etal-2019-joey,
    title = "Joey {NMT}: A Minimalist {NMT} Toolkit for Novices",
    author = "Kreutzer, Julia  and
      Bastings, Jasmijn  and
      Riezler, Stefan",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP): System Demonstrations",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D19-3019",
    doi = "10.18653/v1/D19-3019",
    pages = "109--114",
}
```

## Naming
Joeys are [infant marsupials](https://en.wikipedia.org/wiki/Marsupial#Early_development). 

