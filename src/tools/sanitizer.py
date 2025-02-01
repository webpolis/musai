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
import argparse
from pathlib import Path
from typing import Tuple, Optional

from loguru import logger
from music21 import converter, note, chord
from music21.stream.base import Score
from tqdm import tqdm


def trim_midi(score: Score) -> Tuple[Optional[int], Optional[int]]:
    """
    Identifies the first and last measures containing non-rest elements.
    Returns (None, None) if no non-rest elements are found.
    """
    start_measure = None
    end_measure = None

    for element in score.flatten().elements:
        if isinstance(element, (note.Note, chord.Chord, note.Unpitched)):
            current_measure = element.measureNumber
            if start_measure is None:
                start_measure = current_measure
            end_measure = current_measure  # Update to last found non-rest
        elif isinstance(element, note.Rest) and element.duration.quarterLength > 0:
            continue  # Skip rests but check other elements

    return start_measure, end_measure


def sanitize_filename(filename: str) -> str:
    """Sanitizes filename by replacing non-alphanumeric characters with underscores."""
    name, ext = os.path.splitext(filename)
    sanitized = re.sub(r'[^a-z\d\-]', '_', name.lower())
    sanitized = re.sub(r'_+', '_', sanitized).strip('_')
    return f"{sanitized}{ext}" if sanitized else f"untitled{ext}"


def main():
    parser = argparse.ArgumentParser(
        description='Process MIDI files for cleaning and trimming.')
    parser.add_argument('-m', '--midis_path', required=True,
                        help='Directory containing MIDI files or a text file listing MIDI paths')
    parser.add_argument('-g', '--glob_pattern', default='*.mid',
                        help='Glob pattern for MIDI file discovery')
    parser.add_argument('-o', '--output_dir', default='processed',
                        help='Output directory for processed MIDI files')
    parser.add_argument('--no-rename', dest='rename', action='store_false',
                        help='Disable filename sanitization')
    parser.add_argument('--no-trim', dest='trim', action='store_false',
                        help='Disable silence trimming from MIDI boundaries')
    args = parser.parse_args()

    # Configure paths and ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    midis_path = Path(args.midis_path)

    # Collect MIDI file paths
    if midis_path.is_file():
        with open(midis_path) as f:
            midi_paths = [Path(line.strip()) for line in f if line.strip()]
    else:
        midi_paths = list(midis_path.glob(args.glob_pattern))

    # Process each MIDI file
    for midi_path in tqdm(midi_paths, desc='Processing MIDI files'):
        try:
            score = converter.parse(str(midi_path))
        except Exception as e:
            logger.error(f"Failed to parse {midi_path.name}: {str(e)}")
            continue

        # Filename sanitization
        fname = sanitize_filename(
            midi_path.name) if args.rename else midi_path.name
        output_path = output_dir / fname

        # Score trimming logic
        if args.trim:
            start, end = trim_midi(score)
            if start is not None and end is not None and start <= end:
                try:
                    score = score.measures(start, end)
                    name_part, ext_part = os.path.splitext(fname)
                    output_path = output_dir / f"{name_part}_trim{ext_part}"
                except Exception as e:
                    logger.warning(f"Trimming failed for {
                                   midi_path.name}: {str(e)}")
            else:
                logger.info(f"No trimming needed for {midi_path.name}")

        # Write processed MIDI file
        try:
            score.write('midi', fp=str(output_path))
        except Exception as e:
            logger.error(f"Failed to write {output_path.name}: {str(e)}")


if __name__ == "__main__":
    main()
