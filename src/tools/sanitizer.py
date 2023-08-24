"""
MusAI

Author: Nicolás Iglesias
Email: nfiglesias@gmail.com

This file is part of MusAI, a project for generating MIDI music using 
a combination of machine learning algorithms.

This script offers optional methods to sanitize a set of MIDI files.

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
import os
import re
import ray
import argparse
from loguru import logger
from pathlib import Path
from music21 import converter, note, chord
from music21.stream.base import Score
from tqdm import tqdm
from tokenizer import deco, ProgressBar, ProgressBarActor


def trim_midi(score: Score):
    start_measure = None
    end_measure = 0

    for element in score.flatten().elements:
        if isinstance(element, note.Note) or \
                isinstance(element, note.Rest) or \
            isinstance(element, note.Unpitched) or \
                isinstance(element, chord.Chord):
            if start_measure is None and not element.isRest:
                start_measure = element.measureNumber
            if not element.isRest and element.measureNumber > end_measure:
                end_measure = element.measureNumber

    return start_measure, end_measure


if __name__ == "__main__":
    # parse command line arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-m', '--midis_path', default=None,
                            help='The path where MIDI files can be located', type=str)
    arg_parser.add_argument('-g', '--midis_glob', default='*.mid',
                            help='The glob pattern used to locate MIDI files', type=str)
    arg_parser.add_argument('-o', '--output_path', default='out',
                            help='The path where the sanitized MIDI files will be saved', type=str)
    arg_parser.add_argument('-n', '--rename', help='Sanitize filename for convenience',
                            action='store_true', default=True)
    arg_parser.add_argument('-t', '--trim', help='Remove silence from beginning and end of MIDI songs',
                            action='store_true', default=True)
    args = arg_parser.parse_args()

    Path(args.output_path).mkdir(parents=True, exist_ok=True)

    if os.path.isfile(args.midis_path):
        midi_file_paths = [line.strip()
                           for line in open(args.midis_path) if line.strip()]
    else:
        midi_file_paths = list(Path(args.midis_path).glob(args.midis_glob))

    for midi_path in tqdm(midi_file_paths):
        midi_score = converter.parse(midi_path)
        fname = Path(midi_path).name

        if args.rename:
            # rename
            fname = re.sub(r'[^a-z\d\.]{1,}', '_', fname.lower(), flags=re.IGNORECASE)

        if args.trim:
            # trim MIDI
            start_measure, end_measure = trim_midi(midi_score)
            trimmed_score = midi_score.measures(start_measure, end_measure)
            trim_path = re.sub(r'([\w\d_\-]+)(\.[a-z]+)$', '\\1_trim\\2', fname)
            trim_path = f'{args.output_path}/{trim_path}'

            trimmed_score.write('midi', fp=trim_path)
