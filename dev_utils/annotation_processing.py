import pandas as pd
import numpy as np

def standardize(annotations, sep=',', labels='auto'):
    """ Standardize the annotation table format.

        The input table can be passed as a pandas DataFrame or as the filename of a csv file.

        The table headings are renamed to conform with the ketos standard naming convention, following the 
        name mapping specified by the user. 

        The label mapping is stored as a class attribute named 'label_dict' within the output table 
        and may be retrieved with `df.attrs['label_dict']`.

        Required Columns:
            - 'filename': The name or path of the file associated with each annotation.
            - 'label': The label or category associated with each annotation. If `unfold_labels` is True,
                    this column may contain multiple labels separated by `label_sep`.

            Optional Columns (depending on usage):
            - 'start': The start time or position of the annotation.
            - 'end': The end time or position of the annotation.

        Args:
            annotations: str, pandas DataFrame
                If a string, it is assumed to be the path to a CSV file containing the annotation table.
                If a pandas DataFrame, it is used directly as the annotation table.
            sep: str
                Separator. Only relevant if filename is specified. Default is ",".
            labels: 'auto', None, dict, or list
                - 'auto' (default): All unique labels in the table are automatically mapped to integers starting from 0.
                - None: No label mapping is applied, labels are left as-is.
                - dict: A user-specified mapping of labels to integers (Note that ketos expects labels to be incremental integers starting with 0).
                - list: A subset of labels to map to integers starting from 0.
                Any unspecified label is mapped to -1.

        Returns:
            df: pandas DataFrame
                Standardized annotation table
        
        Note:
            The function assumes that the necessary preprocessing (e.g., renaming columns to match the expected names) 
            has been done prior to calling this function. It is the responsibility of the user to ensure that the input 
            table conforms to the expected format.
    """
    # Determine if 'data' is a file path or a DataFrame
    if isinstance(annotations, str):
        df = pd.read_csv(annotations, sep=sep)
    elif isinstance(annotations, pd.DataFrame):
        df = annotations.copy()
    else:
        raise ValueError("Annotations must be a pandas DataFrame or a file path to a CSV file.")

    # Ensure the DataFrame contains 'filename' and 'label' columns
    missing_columns = [col for col in ['filename', 'label'] if col not in df.columns]
    if missing_columns:
        raise AssertionError(f"Missing required column(s): {', '.join(missing_columns)}")
    
    label_mapping = dict()
    # Handle label mapping based on the labels argument
    if labels == 'auto':
        unique_labels = sorted(pd.unique(df['label']))
        label_mapping = {label: i for i, label in enumerate(unique_labels)}
        df['label'] = df['label'].map(label_mapping).fillna(-1).astype(int)
    elif isinstance(labels, list):
        label_mapping = {label: i for i, label in enumerate(labels)}
        df['label'] = df['label'].map(label_mapping).fillna(-1).astype(int)
    elif isinstance(labels, dict):
        label_mapping = labels
        df['label'] = df['label'].map(label_mapping).fillna(-1).astype(int)
    elif labels is not None:
        raise ValueError("Unsupported value for labels argument. Use 'auto', None, a list of labels, or a dict.")

    # always sort by filename (first) and start time (second)
    by = ['filename']
    if 'start' in df.columns.values: 
        by += ['start']
    df.sort_values(by=by, inplace=True, ignore_index=True)

    # store label dictionary as class attribute
    df.attrs["label_dict"] = label_mapping
    
    return df

def define_segments(df, duration=3.0, center=True):
    """
    Calculates new start and end times for a segment. 
    
    Args:
        df (pd.DataFrame): DataFrame containing 'start' and 'end' columns.
        duration (float): Duration of the new segment (default is 3.0 seconds).
        center (bool): Whether to centralize the segment or place it randomly (default is True).
        
    Modifies:
        df (pd.DataFrame): Updates 'start' and 'end' columns with new values.
    """

    df = df.copy()
    
    if center:
        # Centralize the segment
        segment_midpoint = (df['start'] + df['end']) / 2
        df.loc[:, 'start'] = np.maximum(0, segment_midpoint - duration / 2)
    else:
        # Randomly place the interval within the original bounds
        max_start_time = np.maximum(0, df['end'] - duration)
        df.loc[:, 'start'] = np.random.uniform(df['start'], max_start_time)
    
    # Update the 'end' column based on the new 'start' column and the duration
    df.loc[:, 'end'] = df['start'] + duration

    return df

def generate_time_shifted_segments(
        df: pd.DataFrame, 
        step: float, 
        min_overlap: float = 0.5, 
        duration: float | None = None, 
        include_unshifted: bool = True
    ) -> pd.DataFrame:
    """
    Generate multiple time-shifted segments of each annotation by shifting the selection window
    in steps of length `step` both forward and backward in time, ensuring a minimum overlap
    with the *original* annotation.

    Note: The shifting process can result in `start` and `end` times that fall outside the original 
    recording. For example, if an annotation starts at time 5 and a shift of -10 is applied, 
    the new start time will be -5. Similarly, if an annotation ends at 10 and a shift results 
    in a duration that extends beyond the recording length (e.g., to time 15 when the recording 
    ends at 12) the end time will be 15. Users should handle these cases as needed.
    

    Args:
        df : pd.DataFrame
            A DataFrame containing at least 'start' and 'end' columns, which define 
            the intervals for each annotation.
        step : float
            The step size (in seconds) for shifting the window both forward and backward 
            directions.
        min_overlap : float
            The minimum overlap fraction (between 0 and 1) required between the 
            *original* annotation and the shifted annotation. This fraction is relative
            to the duration of the *original* annotation.
            - Must be greater than 0 and less than or equal to 1.
            - 1 means the entire duration of the original interval must be contained
              within the shifted interval.
        duration : float, optional
            The duration of each shifted instance. If None, the original annotation's 
            duration is used for each shifted instance. Defaults to None.
        include_unshifted : bool, optional
            If True, includes the unshifted segment annotation in the output (the central 
            annotation is without shift and often corresponds to just the annotation 
            itself, if the annotation and duration have the same length). If False, excludes the original annotation from the generated shifted instances. Defaults to False.

    Returns:
        pd.DataFrame
            A new DataFrame containing the time-shifted instances. Preserves all other columns.

    Raises:
        ValueError
            If `min_overlap` is not within the range [0, 1].

    Example:
        >>> import pandas as pd
        >>> data = {'start': [0, 10], 'end': [5, 15]}
        >>> df = pd.DataFrame(data)
        >>> generate_time_shifted_segments(df, step=2.5, min_overlap=0.5, duration=5.0, include_unshifted=False)
           start   end
        0   -2.5   2.5
        0    2.5   7.5
        1    7.5  12.5
        1   12.5  17.5

        >>> generate_time_shifted_segments(df, step=1.0, min_overlap=0.5, include_unshifted=True)
           start  end
        0     -2    3
        0     -1    4
        0      0    5
        0      1    6
        0      2    7
        1      8   13
        1      9   14
        1     10   15
        1     11   16
        1     12   17
    """
    # Check if min_overlap is within the valid range [0, 1]
    if not (0 < min_overlap <= 1):
        raise ValueError("min_overlap must be greater than 0 and less than or equal to 1.")

    # Initialize list to store all time-shifted instances
    shifted_instances = []
    
    for _, row in df.iterrows():
        original_start, original_end = row['start'], row['end']
        annotation_duration = original_end - original_start
        
        # Use the annotation's original duration if duration is None
        current_duration = duration if duration is not None else annotation_duration

        # Determine the minimum required overlap based on the shorter duration
        # min_overlap_duration = min(annotation_duration, current_duration) * min_overlap
        min_overlap_duration = annotation_duration * min_overlap

        # Create shifted windows both forward and backward in time
        shift_range = np.arange(-annotation_duration, annotation_duration, step)

        # Optionally exclude shift == 0 to avoid duplicating the original annotation
        if not include_unshifted:
            shift_range = shift_range[shift_range != 0]  # Exclude 0 from the shift range

        for shift in shift_range:
            # Calculate the shifted start and end times
            new_start = original_start + shift
            new_end = new_start + current_duration

            # Ensure overlap between shifted window and original annotation
            overlap_start = max(original_start, new_start)
            overlap_end = min(original_end, new_end)
            overlap_duration = max(0, overlap_end - overlap_start)

            if overlap_duration >= min_overlap_duration:
                # Create a new instance if overlap condition is satisfied
                new_instance = row.copy()
                new_instance['start'] = new_start
                new_instance['end'] = new_end
                shifted_instances.append(new_instance)
    
    # Create a new DataFrame from the shifted instances
    shifted_df = pd.DataFrame(shifted_instances)
    
    return shifted_df

def create_random_segments(files, duration, num, label=0, annotations=None, buffer=0, max_attempts_per_file=5):
    """
    Generates a specified number of random audio segments from a list of files, ensuring no overlap with annotations.

    Args:
        files (pd.DataFrame): DataFrame with file information, must include 'filename' and 'duration'.
        duration (float): Length of each audio segment to generate in seconds.
        num (int): Number of audio segments to generate.
        label (int): Value to be assigned to the created selections.
        annotations (pd.DataFrame, optional): DataFrame containing annotations to avoid, with columns 'filename', 'start', and 'end'.
        buffer (float): Buffer time in seconds to add around each segment to avoid annotations.
        max_attempts_per_file (int): Maximum number of attempts to generate a segment per file.

    Returns:
        pd.DataFrame: DataFrame containing generated segments with columns 'filename', 'start', and 'end'.

    Raises:
        Warning: If the number of generated samples is less than the requested number due to overlap or file duration limits.
    """    
    results = []

    # Filter files that are too short for the required duration and buffer
    valid_files = files[files['duration'] >= duration + 2 * buffer].copy()

    # If no valid files remain, we can't generate any segments
    if valid_files.empty:
        print("Warning: No files are long enough to generate the required segments with the specified duration and buffer.")
        return pd.DataFrame(results)

    # Normalize file durations to create a weighted sampling probability
    valid_files['prob'] = valid_files['duration'] / valid_files['duration'].sum()


    # Attempt to generate the desired number of segments
    while len(results) < num:
        # Try to generate a valid segment from a file
        for _ in range(max_attempts_per_file):
            # Randomly sample a file based on weighted probabilities
            chosen_file = valid_files.sample(1, weights='prob').iloc[0]
            max_start = chosen_file['duration'] - duration - buffer
            
            # If no valid start time exists, skip this file
            if max_start <= 0:
                continue

            # Randomly choose a start time within the valid range
            start = buffer + np.random.uniform(0, max_start)
            end = start + duration

            # Validate the chosen segment against annotations (if provided)
            if validate_segment(annotations, chosen_file['filename'], start, end, buffer):
                results.append({
                    'filename': chosen_file['filename'],
                    'label': label,
                    'start': start,
                    'end': end
                })
                break  # Break the retry loop once a valid segment is found

    return pd.DataFrame(results)


def validate_segment(annotations, filename, start, end, buffer):
    """
    Validates if a segment is free from overlap with existing annotations.

    Args:
        annotations (pd.DataFrame): DataFrame with 'filename', 'start', and 'end' columns.
        filename (str): The name of the file the segment is being generated from.
        start (float): The start time of the generated segment.
        end (float): The end time of the generated segment.
        buffer (float): Buffer time in seconds to avoid overlaps.

    Returns:
        bool: True if the segment does not overlap with any existing annotations, otherwise False.
    """
    if annotations is None:
        return True  # No annotations to validate against

    # Filter annotations by the filename
    file_annotations = annotations[annotations['filename'] == filename]

    # Check if the segment overlaps with any annotation
    for _, annot in file_annotations.iterrows():
        # The condition checks if there is no overlap by testing two scenarios:
        # 1. (end + buffer <= annot['start']): The segment ends before the annotation starts (including buffer).
        # 2. (start - buffer >= annot['end']): The segment starts after the annotation ends (including buffer).
        #
        # If neither condition is true (i.e., both are False), it means the segment overlaps with the annotation.
        # Therefore, we negate the condition with "not" to detect the overlap.
        if not (end + buffer <= annot['start'] or start - buffer >= annot['end']):
            return False  # Overlaps with annotation

    return True  # No overlap found