#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
create_db.py

Dataset creation utility for acoustic classification and detection tasks.
Generates fixed-length spectrogram representations from continuous audio
recordings and saves them as images, organized by label. Supports both
annotation-based segment extraction and random background sampling, with
optional time-shift augmentation.

Author
------
Bruno Padovese (HALLO Project, SFU)
https://github.com/bpadovese
"""

# =============================================================================
# Imports
# =============================================================================

import numpy as np
import os
import pandas as pd
import json
import librosa
import random
from tqdm import tqdm
from pathlib import Path
from dev_utils.audio_processing import load_segment
from dev_utils.annotation_processing import standardize, define_segments, generate_time_shifted_segments, create_random_segments
from data_handling.spec_preprocessing import classifier_representation
from dev_utils.file_management import file_duration_table

def create_db(data_dir, audio_representation, annotations=None, annotation_step=0, step_min_overlap=0.5, labels=None, 
              output=None, random_selections=None, avoid_annotations=None, overwrite=False, seed=None, 
              n_samples=None, only_augmented=False):
    """
    Create a database of spectrogram images from audio files and annotations.

    This function reads annotations or generates random segments to extract audio
    clips from the provided data directory. For each audio segment, it generates
    a spectrogram representation and saves it as an image in a label-specific
    subdirectory.

    Parameters
    ----------
    data_dir : str
        Path to the directory containing the audio files.
    audio_representation : str
        Path to the JSON file specifying spectrogram parameters (window, step, sr, etc.).
    annotations : str, optional
        Path to the CSV file with labeled annotations.
    annotation_step : float, optional
        Step size (in seconds) for generating time-shifted augmented samples.
    step_min_overlap : float, optional
        Minimum overlap ratio between the annotation and the shifted window.
    labels : dict or list, optional
        Label mapping or list of labels to include.
    output : str, optional
        Output folder to save generated spectrogram images.
    random_selections : tuple, optional
        Tuple (num_segments, label, optional_filename_list) to generate random background samples.
    avoid_annotations : str, optional
        Path to a CSV file containing annotations to avoid when generating random samples.
    overwrite : bool, optional
        Whether to overwrite the output folder if it exists.
    seed : int, optional
        Random seed for reproducibility.
    n_samples : int, optional
        Limit the number of samples per label. Useful for testing.
    only_augmented : bool, optional
        If True, only include augmented (time-shifted) samples.
    """

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    if random_selections is None and annotations is None:
        raise Exception("Missing value: Either annotations or random_selection must be defined.") 

    # Open and read the audio configuration file (e.g., JSON file with audio settings)
    with open(audio_representation, 'r') as f:
        config = json.load(f)

    print('Extracting files durations...')
    files = file_duration_table(data_dir, num=None)
    # Creating a dictionary for quick and easy lookup and access: {filename: duration}
    file_durations = dict(zip(files["filename"], files["duration"]))

    selections = {}
    annots = None

    # ---------------------------------------------------------------------
    # Process annotation-based selections
    # ---------------------------------------------------------------------
    if annotations: # If an annotation table is provided
        annots = pd.read_csv(annotations)
        annots = standardize(annots, labels=labels) # Standardize annotations by mapping labels to integers
        labels = [l for l in annots.label.unique() if l != -1] # Unique labels in the annotations, excluding -1

        # Check if start and end times are present in the annotation dataframe
        if 'start' in annots.columns and 'end' in annots.columns:
            for label in labels:
                # Define segments for the given label based on annotation data
                selections[label] = define_segments(annots, duration=config['duration'], center=True)

                # If annotation_step is set, create time-shifted instances
                if annotation_step > 0:
                    shifted = generate_time_shifted_segments(
                        selections[label], step=annotation_step, 
                        min_overlap=step_min_overlap, include_unshifted=False
                    )              
                    selections[label] = shifted if only_augmented else pd.concat([selections[label], shifted], ignore_index=True)
            
            # Filter out invalid annotations
            selections[label] = selections[label][
                (selections[label]['start'] >= 0) &  # Start time must be non-negative
                (selections[label].apply(lambda row: row['end'] <= file_durations.get(row['filename'], float('inf')), axis=1))  # End time must be within file duration
            ]

        else:
            # If start and end are not present, treat annotations as selections directly 
            for label in labels:
                selections[label] = annots.loc[annots['label'] == label]

    # ---------------------------------------------------------------------
    # Handle random background selections
    # ---------------------------------------------------------------------
    if random_selections is not None: 
        num_segments = random_selections[0] # Number of segments to generate
        if avoid_annotations is not None and annotations is None: # Avoid areas with existing annotations
            annots = pd.read_csv(avoid_annotations)
            annots = standardize(annots, labels=labels)
            
            if num_segments == 'same':
                raise ValueError("The number of background samples to generate cannot be 'same' when avoid_annotations is being used.")

        if num_segments == 'same': # If num_segments is 'same', generate as many samples as the largest selection
            biggest = max(len(v) for v in selections.values())
            num_segments = biggest


        print(f'\nGenerating {num_segments} samples with label {random_selections[1]}...')
        # If filenames are provided, filter the file list based on them
        if random_selections[2]:
            with open(random_selections[2], 'r') as file:
                subset_filenames = file.read().splitlines()
            files = files[files['filename'].isin(subset_filenames)]
        
        # Generate random segments based on the file durations and label
        rando = create_random_segments(files, config['duration'], num_segments, 
                                       label=random_selections[1], annotations=annots)

        
        if labels is None:
            labels = []

        if random_selections[1] in labels: 
            # if the random selection label already exists in the selections, concatenate the generatiosn with the selections that already exist
            selections[random_selections[1]] = pd.concat([selections[random_selections[1]], rando], ignore_index=False) # concatenating the generated random selections with the existings selections
        else:
            # if the random selections label did not yet exist in the selections, add it to the list of labels
            labels.append(random_selections[1])
            selections[random_selections[1]] = rando

    # ---------------------------------------------------------------------
    # Prepare output directory
    # ---------------------------------------------------------------------
    if output is None:
        output = os.path.join('images', 'dataset_images')

    output_path = Path(output)
    if overwrite and output_path.exists():
        print(f"Overwriting existing directory: {output_path}")
        for item in output_path.glob("*"):
            if item.is_file():
                item.unlink()
                
    output_path.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # Generate and save spectrogram images
    # ---------------------------------------------------------------------
    print('\nSaving spectrograms as images...')
    for label in labels:
        label_dir = output_path / str(label)
        label_dir.mkdir(parents=True, exist_ok=True)
        print(f'\nProcessing label {label}...')

        selections_label = selections[label]
        if n_samples is not None:
            selections_label = selections_label.sample(n=n_samples)

        file_image_counts = {}

        for _, row in tqdm(selections_label.iterrows(), total=selections_label.shape[0]):
            start = row.get('start', 0)
            file_path = os.path.join(data_dir, row['filename'])
            file_duration = librosa.get_duration(path=file_path)
            if start >= file_duration:
                start = max(0, file_duration - config['duration'] / 2)
            end = start + config['duration']

            y, sr = load_segment(path=file_path, start=start, end=end, new_sr=config['sr'])
            spectrogram_image = classifier_representation(
                y, config["window"], config["step"], sr, config["num_filters"], 
                fmin=config["fmin"], fmax=config["fmax"]
            )
            
            base_filename = Path(row['filename']).stem
            idx = file_image_counts.get(base_filename, 0) + 1
            output_path_img = label_dir / f"{base_filename}_{idx}.png"
            spectrogram_image.save(output_path_img)

def main():
    import argparse

    class ParseKwargs(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            if len(values) == 1 and not '=' in values[0]:  # Single value
                setattr(namespace, self.dest, [int(values[0])] if values[0].isdigit() else [values[0]])
            elif not any('=' in value for value in values):  # List of values
                setattr(namespace, self.dest, [int(val) if val.isdigit() else val for val in values])
            else:  # Key-value pairs
                kwargs = {}
                for value in values:
                    if '=' not in value:
                        parser.error(f"Invalid format for {option_string}: expected key=value but got '{value}'")
                    key, val = value.split('=')
                    if val.isdigit():
                        val = int(val)
                    kwargs[key] = val
                setattr(namespace, self.dest, kwargs)
                
    class RandomSelectionsAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            if len(values) < 2:
                parser.error("--random_selections requires at least two arguments")
            x = values[0] if values[0] == 'same' else int(values[0])
            y = int(values[1])
            z = values[2] if len(values) > 2 else None
            setattr(namespace, self.dest, (x, y, z))

    parser = argparse.ArgumentParser(description="Generate spectrogram images from audio files.")
    parser.add_argument('data_dir', type=str, help='Path to the directory containing the audio files')
    parser.add_argument('audio_representation', type=str, help='Path to the audio representation config file')
    parser.add_argument('--annotations', default=None, type=str, help='Path to the annotations .csv')
    parser.add_argument('--annotation_step', default=0, type=float, help='Produce multiple time shifted representations views for each annotated  section by shifting the annotation  \
                window in steps of length step (in seconds) both forward and backward in time. The default value is 0.')
    parser.add_argument('--step_min_overlap', default=0.5, type=float, help='Minimum required overlap between the annotated section and the representation view, expressed as a fraction of whichever of the two is shorter. Only used if step > 0.')
    parser.add_argument('--labels', default=None, nargs='*', action=ParseKwargs, help='Specify a label mapping. Example: --labels background=0 upcall=1 will map labels with the string background to 0 and labels with string upcall to 1. \
        Any label not included in this mapping will be discarded. If None, will save every label in the annotation csv and will map the labels to 0, 1, 2, 3....')
    parser.add_argument('--random_selections', default=None, nargs='+', type=str, action=RandomSelectionsAction, help='Will generate random x number of samples with label y. By default, all files in the data_dir and subdirectories will be used.  \
                        To limit this, pass a .txt file with the list of filenames relative to data/dir to sample from. --random_selections x y [filelist.txt].')
    parser.add_argument('--n_samples', default=None, type=int, help='randomly select n samples from the annotations')
    parser.add_argument('--avoid_annotations', default=None, type=str, help="Path to .csv file with annotations of upcalls to avoid. Only used with --random_selections. If the annotations option is being used, this argument is ignored.")
    parser.add_argument('--output', default=None, type=str, help='HDF5 dabase name. For isntance: db.h5')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite the database. Otherwise append to it')
    parser.add_argument('--seed', default=None, type=int, help='Seed for random number generator')
    parser.add_argument('--only_augmented', action='store_true', help='Use only augmented segments (no originals).')
    args = parser.parse_args()

    create_db(
        data_dir=args.data_dir,
        audio_representation=args.audio_representation,
        annotations=args.annotations,
        annotation_step=args.annotation_step,
        step_min_overlap=args.step_min_overlap,
        labels=args.labels,
        output=args.output,
        random_selections=args.random_selections,
        avoid_annotations=args.avoid_annotations,
        overwrite=args.overwrite,
        seed=args.seed,
        n_samples=args.n_samples,
        only_augmented=args.only_augmented
    )

if __name__ == "__main__":
    main()