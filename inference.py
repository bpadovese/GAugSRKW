#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inference.py

Performs segment-wise inference on continuous audio recordings using a trained
deep learning classifier. Audio is divided into fixed-length segments, converted
to Mel-spectrogram representations, and passed through the model to generate
probabilistic detections. Outputs per-segment predictions as a CSV file.

Author
------
Bruno Padovese (HALLO Project, SFU)
https://github.com/bpadovese
"""

# =============================================================================
# Imports
# =============================================================================

import numpy as np
import soundfile as sf
import json
import librosa
import pandas as pd
from tqdm import tqdm
from torch.nn.functional import sigmoid
from torchvision import transforms
from lightning.fabric import Fabric
from pathlib import Path
from data_handling.dataset import ConditionalResize
from data_handling.spec_preprocessing import classifier_representation
from dev_utils.nn import resnet18_for_single_channel

# =============================================================================
# Helper Functions
# =============================================================================

def load_file_list(file_list_path):
    """
    Load file paths from a .txt or .csv file into a list.

    Parameters
    ----------
    file_list_path : str or Path
        Path to a text or CSV file with one filename per line.

    Returns
    -------
    list[str]
        List of file paths.
    """
    file_list_path = Path(file_list_path)
    if file_list_path.suffix in {".txt", ".csv"}:
        return pd.read_csv(file_list_path, header=None).iloc[:, 0].tolist()
    else:
        raise ValueError("Unsupported file list format. Use .txt or .csv.")

# =============================================================================
# Core Functions
# =============================================================================

def process_audio(file_path, config, model, input_shape):
    """
    Run inference on a single audio file by segmenting it into fixed-length
    chunks, computing their audio representation, and passing them
    through the trained model.

    Parameters
    ----------
    file_path : Path
        Path to the input audio file.
    config : dict
        Dictionary containing audio representation parameters.
    model : torch.nn.Module
        Trained neural network model.
    input_shape : tuple(int, int)
        Expected input shape (H, W) for the model.

    Returns
    -------
    predictions_data : dict
        Dictionary containing filenames, start/end times, and predicted scores.
    """

    predictions_data = {
        "filename": [],
        "start": [],
        "end": [],
        "prediction": [], 
    }

    # Open the audio file with soundfile
    with sf.SoundFile(file_path) as audio_file:
        sr = audio_file.samplerate
        segment_length = int(config['duration'] * sr)  # Segment duration in samples
        total_segments = int(np.ceil(audio_file.frames / segment_length))

        with tqdm(total=total_segments, desc=f"Processing Segments for {file_path}", leave=False) as segment_pbar:
            for i in range(total_segments):
                start_sample = i * segment_length
                end_sample = min(start_sample + segment_length, audio_file.frames)  # Ensure bounds are not exceeded
                start_time = start_sample / sr
                end_time = end_sample / sr

                # ignore tiny segments
                if end_time - start_time < 0.01:
                    continue

                # Read current segment and pad if shorter than required
                audio_file.seek(start_sample) 
                segment = audio_file.read(end_sample - start_sample)
                if len(segment) < segment_length:
                    segment = np.pad(segment, (0, segment_length - len(segment)), mode='reflect')

                # Resample if needed
                if config['sr'] is not None and config['sr'] != sr:
                    segment = librosa.resample(segment, orig_sr=sr, target_sr=config['sr'])
                
                # Convert the audio segment to the model's expected representation
                representation_data = classifier_representation(
                    segment, config["window"], config["step"], config['sr'], config["num_filters"], 
                    fmin=config["fmin"], fmax=config["fmax"]
                )

                transform_pipeline = transforms.Compose([
                    ConditionalResize(input_shape),
                    transforms.ToTensor(),
                ])
                
                processed_segment = transform_pipeline(representation_data)

                # Add batch dimension for model input
                input_tensor = processed_segment.unsqueeze(0)  # (1, C, H, W)

                # Perform inference (forward pass)
                logits = model(input_tensor)
                probabilities = sigmoid(logits).detach().cpu().numpy()

                # Append to predictions list
                predictions_data["filename"].append(file_path.relative_to(config['audio_data']))
                predictions_data["start"].append(start_time)
                predictions_data["end"].append(end_time)
                predictions_data["prediction"].append(probabilities.squeeze()) 
            
                segment_pbar.update(1)

    return predictions_data

def main(model_file, audio_data, audio_representation, file_list=None, output_folder=None, input_shape=(128,128)):
    """
    Run inference on all audio files in a directory (or subset specified by file list),
    generating a CSV with segment-level detection probabilities.

    Parameters
    ----------
    model_file : str
        Path to the trained PyTorch model (.pt).
    audio_data : str
        Path to directory containing input audio files.
    audio_representation : str
        Path to the JSON file with spectrogram configuration.
    file_list : str, optional
        Optional .txt/.csv file with subset of filenames to process.
    output_folder : str, optional
        Output folder where detections will be saved.
    input_shape : tuple(int, int)
        Expected input shape for the model.
    """

    model = resnet18_for_single_channel()
    fabric = Fabric()
    state = fabric.load(model_file)
    model.load_state_dict(state["model"])
    model.eval()

    # Setup output
    output_folder = Path(output_folder or '.').resolve()
    output_folder.mkdir(parents=True, exist_ok=True)

    # Load audio representation config
    with open(audio_representation, 'r') as f:
        config = json.load(f)
    audio_path = Path(audio_data)
    config['audio_data'] = audio_path

    # Gather all audio files
    audio_files = [f for ext in ['*.wav', '*.flac'] for f in audio_path.rglob(ext)]
    
    # If a file list is provided, filter audio files accordingly
    if file_list:
        listed_files = [audio_path / Path(f) for f in load_file_list(file_list)]
        listed_paths = {Path(f).resolve() for f in listed_files}
        audio_files = [f for f in audio_files if f.resolve() in listed_paths]

    # Run inference on each audio file
    raw_predictions_list = []
    with tqdm(total=len(audio_files), desc="Processing Audio Files") as pbar:
        for file_path in audio_files:
            file_predictions = process_audio(file_path, config, model, input_shape=input_shape)
            raw_predictions_list.append(file_predictions)
            pbar.update(1)
        
    # Aggregate all predictions
    raw_df = pd.DataFrame(columns=['filename', 'start', 'end', 'score', 'label'])
    for preds in raw_predictions_list:
        scores = []

        for p in preds['prediction']:
            if isinstance(p, (list, np.ndarray)) and len(p) > 1:
                scores.append(float(p[1]))  # class 1 score
            else:
                scores.append(float(p))  # fallback for single-score models

        temp_df = pd.DataFrame({
            'filename': preds['filename'],
            'start': preds['start'],
            'end': preds['end'],
            'score': scores,
            'label': 1  # Assuming binary classification with positive class as 1
        })
        raw_df = pd.concat([raw_df, temp_df], ignore_index=True)

    raw_output = output_folder / "detections_raw.csv"
    raw_df.to_csv(raw_output, index=False)
    print(f"Raw predictions saved to: {raw_output}")

if __name__ == "__main__":
    import argparse
    import ast
    
    def tryeval(val):
        # Literal eval does cast type safely. However, doesnt work for str, therefore the try except.
        try:
            val = ast.literal_eval(val)
        except ValueError:
            pass
        return val
        
    class ParseKwargs(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, dict())
            for value in values:
                key, value = value.split('=')
                getattr(namespace, self.dest)[key] = tryeval(value)

    # parse command-line args
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file', type=str, help='Path to the torch model file (*.pt)')
    parser.add_argument('audio_data', type=str, help='Path to either a folder with audio files.')
    parser.add_argument('audio_representation', type=str, help='Path to the audio representation config file')
    parser.add_argument('--file_list', default=None, type=str, 
                        help='A .csv or .txt file where each row (or line) is the name of a file to detect within the audio folder. \
                        By default, all files will be processed.')
    parser.add_argument('--output_folder', default=None, type=str, 
                        help='Location to output the detections. For instance: detections/')
    parser.add_argument('--input_shape', type=int, nargs='+', default=[128, 128], 
                        help='Input shape as width and height (e.g., --input_shape 128 128).')
    
    args = parser.parse_args()

    if len(args.input_shape) == 1:
        input_shape = (args.input_shape[0], args.input_shape[0])
    elif len(args.input_shape) == 2:
        input_shape = tuple(args.input_shape) #convert to tuple
    else:
        parser.error("--input_shape must be one or two integers.")

    main(args.model_file, args.audio_data, args.audio_representation, 
         file_list=args.file_list, output_folder=args.output_folder, 
         input_shape=input_shape)