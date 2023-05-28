"""
MusAI

Author: Nicolás Iglesias
Email: nfiglesias@gmail.com

This file is part of MusAI, a project for generating MIDI music using 
a combination of machine learning algorithms.

Below is a monolitic script that constructs a corpora of tokens ready to be injected 
into the model. Semantical processing, cleanup and sanitization happens here.

MIT License
Copyright (c) [2023] [Nicolás Iglesias]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import json
import os
import re
import argparse
import ray
from loguru import logger
from pathlib import Path
from miditok import REMIPlus, MMM
from miditok.constants import ADDITIONAL_TOKENS, BEAT_RES, INSTRUMENT_CLASSES
from miditok.utils import merge_tracks_per_class, merge_same_program_tracks, get_midi_programs
from miditoolkit import MidiFile
from tqdm import tqdm

# initialize variables
os.environ['FUNCTION_SIZE_ERROR_THRESHOLD'] = '512'

TOKENS_PATH = '/home/nico/data/ai/models/midi/mix'
MIDIS_PATH = '/home/nico/data/midis/MIDI'
TOKEN_PARAMS_NAME = 'token_params.cfg'
TOKEN_PARAMS_PATH = Path(f'{TOKENS_PATH}/{TOKEN_PARAMS_NAME}')

PITCH_RANGE = range(21, 109)
CLASSES_PERCUSSION = [1, 14, 16]
CLASSES_SYNTHS = [10, 11]
CLASSES_STRINGS = [5, 6]
CLASSES_GUITAR_BASS = [3, 4]
CLASS_REED = [8, 9]
CLASS_EFFECTS = [12, 15]
BINS_VELOCITY = (24)
BINS_TEMPO = (24)

TOKENIZER_ALGOS = ['REMI', 'MMM']

# initialize logger
logger.add('tokenizer_errors_{time}.log', delay=True,
           backtrace=True, diagnose=True, level='ERROR', rotation='10 MB')

# starts orchestration
ray.init()

# define some functions


def to_iterator(obj_ids):
    while obj_ids:
        done, obj_ids = ray.wait(obj_ids)
        yield ray.get(done[0])


@ray.remote
def process_midi(midi_path):
    try:
        midi = MidiFile(midi_path)
    except:
        return None

    programs = get_midi_programs(midi)

    if midi.ticks_per_beat < max(BEAT_RES.values()) * 4:
        return None

    # remove unwanted tracks
    for cls in CLASS_EFFECTS:
        programs_to_delete = list(
            INSTRUMENT_CLASSES[cls]['program_range'])

    try:
        for i in range(0, len(midi.instruments)):
            if midi.instruments[i].program in programs_to_delete:
                del midi.instruments[i]
    except Exception as e:
        return None

    # merge percussion/drums
    merge_tracks_per_class(midi, CLASSES_PERCUSSION)

    # merge synths
    merge_tracks_per_class(midi, CLASSES_SYNTHS)

    # merge strings
    merge_tracks_per_class(midi, CLASSES_STRINGS)

    # merge guitar & bass
    merge_tracks_per_class(midi, CLASSES_GUITAR_BASS)

    # merge_same_program_tracks(midi.instruments)
    midi_name = re.sub(r'[^0-9a-z_]{1,}', '_',
                       str.lower(os.path.basename(midi_path)))

    ray_midi_ref = ray.put({
        'midi': midi,
        'programs': programs,
        'path': midi_path,
        'name': midi_name
    })

    return ray_midi_ref


def get_collection_refs(midis_path=None, midis_glob=None):
    """Pre-process and retrieves a collection of MIDI files, ready for tokenization.

    :return: A dictionary containing a set of {'midi': ..., 'programs': ..., 'path': ...} 
            for each MIDI file in the collection.
    :rtype: dict
    """
    midi_file_paths = list(Path(midis_path).glob(midis_glob))

    logger.info(
        'Processing collection: {coll_size} MIDI files', coll_size=len(midi_file_paths))

    # process MIDIs via Ray
    ray_refs = [process_midi.remote(midi_path)
                for midi_path in midi_file_paths]
    ray_midi_refs = [ref for ref in tqdm(to_iterator(ray_refs))]

    return [ref for ref in ray_midi_refs if ref != None]


def get_tokenizer(params=None, algo='MMM'):
    """Returns a tokenizer.

    :param params: Path to a token_params.cfg file for preloading, defaults to None
    :type params: str, optional
    :return: A MMM or REMIPlus tokenizer.
    :rtype: MIDITokenizer
    """
    if algo not in TOKENIZER_ALGOS:
        raise 'Invalid tokenization algorithm'

    additional_tokens = ADDITIONAL_TOKENS
    additional_tokens['Chord'] = True
    additional_tokens['TimeSignature'] = True
    additional_tokens['Program'] = True
    additional_tokens['nb_tempos'] = BINS_TEMPO
    tokenizer = None

    if algo == 'REMI':
        tokenizer = REMIPlus(pitch_range=PITCH_RANGE,
                             additional_tokens=additional_tokens,
                             nb_velocities=BINS_VELOCITY,
                             params=params)
    elif algo == 'MMM':
        tokenizer = MMM(pitch_range=PITCH_RANGE,
                        additional_tokens=additional_tokens,
                        nb_velocities=BINS_VELOCITY,
                        params=params)

    logger.info('Tokenizer initialized. Using {algo}', algo=algo)

    return tokenizer


# begin program
if __name__ == "__main__":
    # parse command line arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-t', '--tokens_path', default=TOKENS_PATH,
                            help='The output path were tokens are saved', type=str)
    arg_parser.add_argument('-m', '--midis_path', default=MIDIS_PATH,
                            help='The path where MIDI files can be located', type=str)
    arg_parser.add_argument('-g', '--midis_glob', default='*mix*.mid',
                            help='The glob pattern used to locate MIDI files', type=str)
    arg_parser.add_argument('-b', '--bpe', help='Applies BPE to the corpora of tokens',
                            action='store_true', default=False)
    arg_parser.add_argument('-p', '--process', help='Extracts tokens from the MIDI files',
                            action='store_true', default=False)
    arg_parser.add_argument('-s', '--semantical', help='Analyze corpora and process semantical grouping',
                            action='store_true', default=False)
    arg_parser.add_argument('-a', '--algo', help='Tokenization algorithm',
                            choices=TOKENIZER_ALGOS, default='MMM', type=str)
    args = arg_parser.parse_args()

    # initializes tokenizer
    TOKENIZER = get_tokenizer()
    MIDI_COLLECTION_REFS = get_collection_refs(
        args.midis_path, args.midis_glob)
    MIDI_TITLES = [ray.get(ray_midi_ref)['name']
                   for ray_midi_ref in MIDI_COLLECTION_REFS]
    ray_tokenized_refs = None

    if args.process:
        @ray.remote
        def tokenize_set(midi_doc):
            midi = midi_doc['midi']
            programs = midi_doc['programs']

            try:
                tokens = TOKENIZER.midi_to_tokens(
                    midi, apply_bpe_if_possible=args.bpe)

                TOKENIZER.save_tokens(
                    tokens, f"{args.tokens_path}/{midi_doc['name']}.json", programs=programs)
            except:
                return None

            return midi_doc

        logger.info('Processing tokenization: {collection_size} documents', collection_size=len(
            MIDI_COLLECTION_REFS))

        Path(args.tokens_path).mkdir(parents=True, exist_ok=True)

        # process tokenization via Ray
        ray_refs = [tokenize_set.remote(ray_midi_ref)
                    for ray_midi_ref in MIDI_COLLECTION_REFS]
        ray_tokenized_refs = [
            ray_tokenized_ref for ray_tokenized_ref in tqdm(to_iterator(ray_refs))]

        logger.info('Vocab size (no BPE): {vocab_size}',
                    vocab_size=len(TOKENIZER.vocab))
        logger.info('Saving params...')

        """ !IMPORTANT always store the _vocab_base when saving params. 
        Order of keys in the vocab may differ in a new instance of a preloaded TOKENIZER. """
        TOKENIZER.save_params(
            f'{args.tokens_path}/{TOKEN_PARAMS_NAME}', {'_vocab_base': TOKENIZER.vocab})

    if args.bpe:
        # Constructs the vocabulary with BPE, from the tokenized files
        tokens_bpe_path = f'{args.tokens_path}/bpe'
        token_files_paths = [
            f"{args.tokens_path}/{midi_doc['name']}.json" for midi_doc in ray_tokenized_refs if midi_doc != None]

        Path(tokens_bpe_path).mkdir(parents=True, exist_ok=True)

        ray.shutdown()

        if not args.process:
            TOKENIZER = get_tokenizer(
                params=f'{args.tokens_path}/{TOKEN_PARAMS_NAME}')

        logger.info('Learning BPE from vocab size {vocab_size}...', vocab_size=len(
            TOKENIZER.vocab))

        TOKENIZER.learn_bpe(
            vocab_size=int(len(TOKENIZER.vocab)*1.25),
            tokens_paths=token_files_paths,
            start_from_empty_voc=False,
        )

        # Converts the tokenized musics into tokens with BPE
        logger.info('Applying BPE...')
        TOKENIZER.apply_bpe_to_dataset(args.tokens_path, tokens_bpe_path)

        logger.info('Saving params with BPE applied...')
        TOKENIZER.save_params(f'{tokens_bpe_path}/{TOKEN_PARAMS_NAME}')

        logger.info('Vocab size (BPE): {vocab_size}',
                    vocab_size=len(TOKENIZER.vocab))

    if args.semantical:
        logger.info('Semantical processing: {collection_size} documents', collection_size=len(
            MIDI_COLLECTION_REFS.items()))

        token_files_paths = [
            f'{args.tokens_path}/{midi_name}.json' for midi_name in MIDI_TITLES]

        for token_file in tqdm(token_files_paths):
            try:
                tokens = json.load(open(token_file, 'r'))['ids']

                pass
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(e)
