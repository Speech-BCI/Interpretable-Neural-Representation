# Deep Contrastive Learning with Data-driven Representation Embeddings for Interpretable Neural Speech Decoding


## Requirements

To perform training, you should have at least one NVIDIA GPU with a minimum of 20GB of memory. You can reduce memory requirements by lowering the batch size or setting `train_kwargs["precision"] = '16-mixed'`.

You can create a new conda environment and install the required dependencies as follows:

```shell
conda create -n [name] python=3.9 -y
conda activate [name]
conda install pytorch cudatoolkit=11.7 -c pytorch -y
pip install -U -r requirements.txt
```

## Parameters

All training parameters used in the experiments are specified in the `INR_params.yaml` file. For additional details, please refer to the information provided in the paper.

## Preprocessing

**Audio**: The mel-spectrogram transformation of all acoustic signals follows the methods described in the Wavegrad: Estimating gradients for waveform generation paper. The link to the repository is [here](https://github.com/lmnt-com/wavegrad).

For audio VAD (Voice Activity Detection), we used the model from the paper "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations" available on Huggingface's wav2vec2 [here](https://huggingface.co/models).

## Superlet Transform

The superlet transform used for neural signals follows the methods described in the "Time-frequency super-resolution with superlets" paper. The library used is provided by the paper's authors at [this link](https://github.com/TransylvanianInstituteOfNeuroscience/Superlets), and phase calculations were performed using the FieldTrip toolbox available [here](https://www.fieldtriptoolbox.org/).

## Training

We conducted training using PyTorch Lightning. The model was defined using `pytorch-lightning==2.0.4`. The model we used consists of spatial attention, a CNN encoder, and a Transformer encoder. The entire training process, including the model definition, is specified in `NR_model.py` using PyTorch Lightning. You can perform training using the parameters specified in `NR_params.yaml`.
