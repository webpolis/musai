# MusAI

MusAI is an innovative project that leverages the power of machine learning to generate unique and creative MIDI music sequences. With MusAI, you can explore the intersection of art and technology, and unleash your creativity by generating original music compositions.

## Features

- Full-featured tokenizer using parallelization via Ray
- MIDI music generation using a combination of architectures ([RWKV](https://github.com/BlinkDL/RWKV-LM), [VAE](https://en.wikipedia.org/wiki/Variational_autoencoder), etc.)
- Fine-tune or generate a new model from scratch using a custom dataset 
- Instrument based sequence training and generation (drums, bass, etc.)
- Pre-trained embeddings using a Variational Autoencoder ([Experimental](https://github.com/webpolis/musai/wiki/Experimental))
- Adjustable parameters to customize the style and complexity of the generated music
- High-quality output MIDI files for further refinement or direct use in your projects
- Seamless integration with your favorite music production tools via VST bridge (@WIP)

## Installation

`pip install -U -r requirements.txt`

## Usage

The typical workflow is:

- Convert MIDI files into tokens
- Train the model
- Generate new sequences

### [Tokenizer](src/tools/tokenizer.py)


### Usage

```sh
tokenizer.py [-h] [-t TOKENS_PATH] [-m MIDIS_PATH] [-g MIDIS_GLOB] [-b] [-p] [-a {REMI,MMM}] [-c CLASSES]
                    [-r CLASSES_REQ] [-l LENGTH] [-d]

options:
  -h, --help            show this help message and exit
  -t TOKENS_PATH, --tokens_path TOKENS_PATH
                        The output path were tokens are saved
  -m MIDIS_PATH, --midis_path MIDIS_PATH
                        The path where MIDI files can be located
  -g MIDIS_GLOB, --midis_glob MIDIS_GLOB
                        The glob pattern used to locate MIDI files
  -b, --bpe             Applies BPE to the corpora of tokens
  -p, --process         Extracts tokens from the MIDI files
  -a {REMI,MMM}, --algo {REMI,MMM}
                        Tokenization algorithm
  -c CLASSES, --classes CLASSES
                        Only extract these instruments classes (e.g. 1,14,16,3,4,10,11)
  -r CLASSES_REQ, --classes_req CLASSES_REQ
                        Minimum set of instruments classes required (e.g. 1,14,16)
  -l LENGTH, --length LENGTH
                        Minimum sequence length (in beats)
  -d, --debug           Debug mode (disables Ray).

```

#### Instrument Classes

```
CLASS | NAME
------+---------------------
  0   | Piano
  1   | Chromatic Percussion
  2   | Organ
  3   | Guitar
  4   | Bass
  5   | Strings
  6   | Ensemble
  7   | Brass
  8   | Reed
  9   | Pipe
 10   | Synth Lead
 11   | Synth Pad
 12   | Synth Effects
 13   | Ethnic
 14   | Percussive
 15   | Sound Effects  <-- Effects are automatically removed because they don't introduce 
                           relevant information to the model.
 16   | Drums
```

### [Trainer](src/tools/trainer.py)

### (optional) Embeddings

> Additionally, you can choose to use a VAE model in replacement of the default architecture's embedding module ([read more](https://github.com/webpolis/musai/wiki/Experimental)).

> There is an extra cost on training performance if you choose to build the VAE embeddings from scratch (using `--vae_emb true`) while training the main model, so it is recommended to train the embeddings beforehand (`--vae_emb train` or see [example](notebooks/vae.ipynb)), but make sure you use the same values for the embeddings size (`--embed_num`) when building the final model.

> Training the final model using pre-trained embeddings (`--vae_emb path_to_pth_file`) will **save** significant _VRAM_.

To train the embeddings alone from scratch, change the arguments to match your needs and run:

```sh
python src/tools/trainer.py -t path_to_tokenized_dataset -o output_path -v train -e 768 -b 24 -p 20 -s 1000 -i 1e-5
```

The saved embedding model will be stored in the output path with a name such as `embvae_#.pth` where `#` is the epoch number. Afterwards, you can use that file as the pre-trained embeddings for training the main and final model, using a command similar to:

```sh
python src/tools/trainer.py -t path_to_tokenized_dataset -o output_path -v path_to_pretrained_embeddings.pth -e 768 -c 2048 -n 12 -b 24 -p 100 -s 1000 -i 1e-5 -g -q
```

You can avoid the first step and let the RWKV architecture train its own embeddings by removing the `-v` or `--vae_emb` option.

### Usage

```sh
trainer.py [-h] [-t TOKENS_PATH] [-o OUTPUT_PATH] [-m BASE_MODEL] [-r LORA_CKPT] [-c CTX_LEN]
                  [-b BATCHES_NUM] [-e EMBED_NUM] [-n LAYERS_NUM] [-p EPOCHS_NUM] [-s STEPS_NUM] [-i LR_RATE]
                  [-d LR_DECAY] [-a] [-l] [-g]

options:
  -h, --help            show this help message and exit
  -t DATASET_PATH, --dataset_path DATASET_PATH
                        The path were tokens parameters were saved by the tokenizer
  -x, --binidx          Dataset is in binidx format (Generated via https://github.com/Abel2076/json2binidx_tool) 
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        The output path were model binaries will be saved
  -m BASE_MODEL, --base_model BASE_MODEL
                        Full path for base model/checkpoint (*)
  -r LORA_CKPT, --lora_ckpt LORA_CKPT
                        Full path for LoRa checkpoint (*)
  -v VAE_EMB, --vae_emb VAE_EMB
                        The pre-trained VAE embeddings. Possible options: 
                        "train" for training alone, from scratch.
                        "train path_to_existing_embeddings.pth" for training alone, from saved model.
                        "true" for training from scratch together with the main model (slow).
                        "path_to_existing_embeddings.pth" to use existing embeddings model while training main model (fast).
  -c CTX_LEN, --ctx_len CTX_LEN
                        The context length
  -b BATCHES_NUM, --batches_num BATCHES_NUM
                        Number of batches
  -e EMBED_NUM, --embed_num EMBED_NUM
                        Size of the embeddings dimension
  -n LAYERS_NUM, --layers_num LAYERS_NUM
                        Number of block layers (*)
  -p EPOCHS_NUM, --epochs_num EPOCHS_NUM
                        Number of epochs
  -s STEPS_NUM, --steps_num STEPS_NUM
                        Number of steps per epoch
  -i LR_RATE, --lr_rate LR_RATE
                        Learning rate. Initial & final derivates from it.
  -d LR_DECAY, --lr_decay LR_DECAY
                        Learning rate decay thru steps
  -a, --attention       Enable tiny attention (*)
  -l, --lora            Activate LoRa (Low-Rank Adaptation) (*)
  -g, --grad_cp         Gradient checkpointing (*)
  -q, --head_qk         Enable head QK (*)

```

_* Only used when training the main model._

### Runner

(@WIP)


## Examples

Check out the [examples](examples/) folder.

## Text Generation

The trainer tool and model included in Musai can be easily reused for training a Large Language Model (LLM) or a Text Generation model of any size. Just provide the corresponding dataset processed with the [binidx](https://github.com/Abel2076/json2binidx_tool) and follow the given instructions.

## Contributing

Contributions to MusAI are welcome! If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
