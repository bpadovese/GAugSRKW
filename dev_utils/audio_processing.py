import numpy as np
import soundfile as sf
import librosa

def load_segment(path, start=None, end=None, new_sr=None, pad='reflect'):
    """
    Loads an audio segment with optional padding.

    Args:
        path (str): Path to the audio file.
        start (float): Start time in seconds.
        end (float): End time in seconds.
        new_sr (int): If provided, resample the audio to this sample rate.
        pad (str): Padding option to apply if the requested segment extends beyond the audio file.
                   Options:
                   - 'zero': Pads the beginning and/or end of the audio with zeros if the start or 
                     end time is outside the file's duration.
                   - 'reflect' (default): Pads the beginning and/or end of the audio with a reflected version 
                     of the audio if the start or end time is outside the file's duration.

    Returns:
        np.ndarray: The loaded audio segment.
        int: The sample rate of the loaded audio.
    """
    # Open the file to get the sample rate and total frames
    with sf.SoundFile(path) as file:
        sr = file.samplerate
        total_frames = file.frames
        file_duration = total_frames / sr  # Duration of the file in seconds

        # Default to the full file if neither start nor end is provided
        if start is None and end is None:
            start, end = 0, file_duration
        
        # Adjust start time if it's negative, and dynamically adjust the end time to maintain duration
        pad_start = 0
        pad_end = 0
        if start is not None and start < 0:
            pad_start = -start  # Calculate how much to pad at the beginning
            start = 0

        # Ensure end time does not exceed the file's duration
        if end is not None and end > file_duration:
            pad_end = end - file_duration  # Calculating how much to pad at the end
            end = file_duration

        # Convert start and end times to frame indices
        start_frame = int(start * sr)
        end_frame = int(end * sr)
        
        # Read the specific segment of the audio file
        file.seek(start_frame)
        audio_segment = file.read(end_frame - start_frame)

    # Apply padding if necessary
    if (pad_start > 0 or pad_end > 0):
        # Calculate padding in frames
        pad_start_frames = int(pad_start * sr)
        pad_end_frames = int(pad_end * sr)

        if pad == 'zero':
            audio_segment = np.pad(audio_segment, (pad_start_frames, pad_end_frames), 'constant')
        elif pad == 'reflect':
            audio_segment = np.pad(audio_segment, (pad_start_frames, pad_end_frames), 'reflect')

    # Resample the audio segment if new sample rate is provided and different from the original
    if new_sr is not None and new_sr != sr:
        audio_segment = librosa.resample(audio_segment, orig_sr=sr, target_sr=new_sr)
        sr = new_sr  
    
    return audio_segment, sr

def get_duration(file_paths):
    """Calculate the durations of multiple audio files.

    Args:
        file_paths (list): List of paths to audio files.

    Returns:
        list: Durations of the audio files in seconds.
    """
    durations = []
    for file_path in file_paths:
        with sf.SoundFile(file_path) as sound_file:
            durations.append(len(sound_file) / sound_file.samplerate)
    return durations