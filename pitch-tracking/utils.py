import mirdata
import crepe
import librosa
import mir_eval
import numpy as np
import matplotlib.pyplot as plt


def compute_dataset_stats(dataset):
    """
    Compute statistics for a given dataset, including the number of tracks,
    distribution by instrument, and distribution by genre.

    Parameters
    ----------
    dataset : mirdata.core.Dataset
        The dataset object for which to compute statistics.

    Returns
    -------
    dict
        A dictionary containing:
        - 'num_tracks' (int): Total number of tracks in the dataset.
        - 'instrument_distribution' (dict): A dictionary where keys are instrument names
          and values are counts of tracks for each instrument.
        - 'genre_distribution' (dict): A dictionary where keys are genre names and values
          are counts of tracks for each genre.

    Notes
    -----
    The function expects that the tracks in the dataset might have attributes named
    'instrument' and 'genre'. If a track lacks these attributes, it's simply not
    considered in the respective distribution.

    Example
    -------
    >>> dataset = mirdata.initialize("medleydb_pitch")
    >>> stats = compute_dataset_stats(dataset)
    >>> print(stats)
    {
        'num_tracks': 100,
        'instrument_distribution': {'flute': 30, 'guitar': 50, ...},
        'genre_distribution': {'jazz': 40, 'rock': 30, ...},
    }
    """
    # YOUR CODE HERE

    # Initialize the dictionaries to store the distributions
    instrument_distribution = {}
    genre_distribution = {}

    # Loop through the tracks in the dataset
    for track_id in dataset.track_ids:
        # Load the track
        track = dataset.track(track_id)

        # Check if the track has an 'instrument' attribute
        if hasattr(track, "instrument"):
            # Increment the count for the instrument in the dictionary
            instrument = track.instrument
            if instrument in instrument_distribution:
                instrument_distribution[instrument] += 1
            else:
                instrument_distribution[instrument] = 1

        # Check if the track has a 'genre' attribute
        if hasattr(track, "genre"):
            # Increment the count for the genre in the dictionary
            genre = track.genre
            if genre in genre_distribution:
                genre_distribution[genre] += 1
            else:
                genre_distribution[genre] = 1

    # Return the statistics
    return {
        "num_tracks": len(dataset.track_ids),
        "instrument_distribution": instrument_distribution,
        "genre_distribution": genre_distribution,
    }


def estimate_pitch(audio_path, voicing_threshold=0.3, use_viterbi=False):
    """
    Estimate the fundamental frequency (pitch) of an audio file using the CREPE algorithm.

    Parameters
    ----------
    audio_path : str
        The file path to the input audio file.
    voicing_threshold : float, optional
        The confidence threshold above which a frame is considered voiced. Frames with confidence
        levels below this threshold are marked as unvoiced (i.e., set to 0 Hz).
        Default is 0.3.
    use_viterbi : bool, optional
        If True, apply Viterbi decoding to smooth the pitch track and obtain more consistent
        pitch estimates over time. Default is False.

    Returns
    -------
    time : np.ndarray
        A 1D numpy array containing time stamps for each frame in seconds.
    frequency : np.ndarray
        A 1D numpy array containing the estimated pitch for each frame in Hz. Unvoiced frames
        are set to 0 Hz.
    confidence : np.ndarray
        A 1D numpy array containing the confidence of the pitch estimate for each frame.
    activation : np.ndarray
        A 2D numpy array representing the activation matrix returned by the CREPE algorithm,
        which can be used to visualize the pitch estimation process.

    """

    # Hint: follow this steps
    # load audio using librosa
    # use crepe.predict
    # you will need to do a little postprocessing before returning the
    # frequency values, remember that you need to determine the voicing first looking at the activation,
    # and then which is the most likely frequency for the voiced frames.
    # read

    # YOUR CODE HERE

    # Load the audio file using librosa
    y, sr = librosa.load(audio_path, sr=None, mono=True)

    # Use the CREPE algorithm to estimate pitch
    time, frequency, confidence, activation = crepe.predict(y, sr, viterbi=use_viterbi)

    # Determine the voicing based on the confidence threshold
    voicing = confidence > voicing_threshold

    # Set the frequency to 0 Hz for unvoiced frames
    frequency[~voicing] = 0

    # Return the time, frequency, confidence, and activation
    return time, frequency, confidence, activation


def evaluate_pitch(data, voicing_threshold=0.3, use_viterbi=False):
    """
    Evaluate pitch estimation for multiple tracks using mir_eval.

    Parameters
    ----------
    data : dict
        Dictionary containing track information. Keyed by track ID with values being track objects.
        Each track object is expected to have an `audio_path` attribute for the audio file and a
        `pitch` attribute which has `times` and `frequencies` attributes.
    voicing_threshold : float, optional
        Threshold on the voicing to determine which frames are unvoiced. Defaults to 0.3.
    use_viterbi : bool, optional
        If True, use the Viterbi algorithm during pitch estimation. Defaults to False.

    Returns
    -------
    dict
        Dictionary containing evaluation scores for each track. Keyed by track ID with values being
        the evaluation results from mir_eval.melody.evaluate (which is a dictionary).

    Notes
    -----
    This function makes use of the `estimate_pitch` function to estimate the pitch for each track,
    and then evaluates the estimated pitch against the ground truth using mir_eval.
    """
    # YOUR CODE HERE

    # Initialize the dictionary to store the evaluation scores
    evaluation_scores = {}

    # Loop through the tracks in the input data
    for track_id, track in data.items():
        # Load the audio file and estimate the pitch
        time, frequency, confidence, _ = estimate_pitch(
            track.audio_path, voicing_threshold, use_viterbi
        )

        # Get the ground truth pitch from the track object
        ref_time = track.pitch.times
        ref_frequency = track.pitch.frequencies

        # Evaluate the estimated pitch using mir_eval
        scores = mir_eval.melody.evaluate(ref_time, ref_frequency, time, frequency)

        # Store the evaluation scores in the dictionary
        evaluation_scores[track_id] = scores

    # Return the evaluation scores
    return evaluation_scores


def prepare_boxplot_data(pitch_scores):
    """
    Prepare pitch tracking evaluation scores for boxplot visualization.

    Parameters
    ----------
    pitch_scores : dict
        A dictionary where the keys are track names and the values are ordered dictionaries
        of evaluation metrics, such as Voicing Recall, Voicing False Alarm, Raw Pitch Accuracy,
        Raw Chroma Accuracy, and Overall Accuracy.

    Returns
    -------
    data_dict : dict
        A dictionary where each key is an evaluation metric and the value is a list of all
        scores for that metric across all tracks. Suitable for use in creating boxplots.

    Examples
    --------
    >>> pitch_scores = {
    ...     'Track_1': OrderedDict([
    ...         ('Voicing Recall', 0.99),
    ...         ('Voicing False Alarm', 0.45),
    ...         ...
    ...     ]),
    ...     'Track_2': OrderedDict([
    ...         ('Voicing Recall', 0.98),
    ...         ('Voicing False Alarm', 0.50),
    ...         ...
    ...     ]),
    ...     ...
    ... }
    >>> data_dict = prepare_boxplot_data(pitch_scores)
    >>> for metric, scores in data_dict.items():
    ...     print(f"{metric}: {scores}")
    Voicing Recall: [0.99, 0.98, ...]
    Voicing False Alarm: [0.45, 0.50, ...]
    ...
    """
    # YOUR CODE HERE

    # Initialize the dictionary to store the data for the boxplot
    data_dict = {}

    # Loop through the pitch scores
    for track_id, scores in pitch_scores.items():
        # Loop through the evaluation metrics
        for metric, value in scores.items():
            # Add the score to the list for the metric
            if metric in data_dict:
                data_dict[metric].append(value)
            else:
                data_dict[metric] = [value]

    # Return the dictionary with the data for the boxplot
    return data_dict


def split_by_instrument(scores_dictionary, tracks_dictionary):
    """
    Split scores by instrument, retaining only the scores for the top 6 most frequently
    occurring instruments.

    The function takes as input a dictionary of scores that have an associated instrument
    and returns a nested dictionary where the outer keys are instrument names and the
    inner dictionaries are scores associated with unique track identifiers.

    Parameters
    ----------
    scores_dictionary : dict
        A dictionary where keys are track IDs and values are objects or dictionaries
        that have an 'instrument' attribute or key and a 'score' attribute or key.
    tracks_dictionary: dict
        Dictionary of mirdata tracks keyed by track_id

    Returns
    -------
    instrument_scores : dict
        A dictionary with instrument names as keys. Each key maps to another dictionary
        containing track IDs as keys and their associated scores as values.

    Example
    --------

    Dictionary should look like:
    {'male singer': {'AClassicEducation_NightOwl_STEM_08': OrderedDict([('Voicing Recall',
                0.9981117230527145),
               ('Voicing False Alarm', 0.46255349500713266),
               ('Raw Pitch Accuracy', 0.9851298190401259),
               ('Raw Chroma Accuracy', 0.9853658536585366),
               ('Overall Accuracy', 0.7301076725130359)]),
                'AClassicEducation_NightOwl_STEM_13': OrderedDict([('Voicing Recall',
                0.995873786407767),
               ('Voicing False Alarm', 0.8500986193293886), ....}
    """
    # YOUR CODE HERE

    # Initialize the dictionary to store the scores by instrument
    instrument_scores = {}

    # Count the occurrences of each instrument
    instrument_counts = {}
    for track_id, track in tracks_dictionary.items():
        if hasattr(track, "instrument"):
            instrument = track.instrument
            if instrument in instrument_counts:
                instrument_counts[instrument] += 1
            else:
                instrument_counts[instrument] = 1

    
    # Sort the instruments by count
    sorted_instruments = sorted(instrument_counts, key=instrument_counts.get, reverse=True)

    # Get the top 6 most frequent instruments
    top_instruments = sorted_instruments[:6]

    # Loop through the scores and assign them to the instrument
    for track_id, score in scores_dictionary.items():
        if hasattr(tracks_dictionary[track_id], "instrument"):
            instrument = tracks_dictionary[track_id].instrument
            if instrument in top_instruments:
                if instrument in instrument_scores:
                    instrument_scores[instrument][track_id] = score
                else:
                    instrument_scores[instrument] = {track_id: score}

    # Return the dictionary with the scores by instrument
    return instrument_scores


def split_by_genre(scores_dictionary, tracks_dictionary):
    """Split scores by genre.

    Parameters
    ----------
    scores_dictionary : dict
        Dictionary of scores keyed by track_id
    tracks_dictionary: dict
        Dictionary of mirdata tracks keyed by track_id

    Returns
    -------
    genre_scores : dict
        Dictionary with genre as keys and a
        dictionary of scores keyed by track_id as values

    Example
    -------

    Dictionary should look like:
    {'Singer/Songwriter': {'AClassicEducation_NightOwl_STEM_08': OrderedDict([('Voicing Recall',
                0.9981117230527145),
               ('Voicing False Alarm', 0.46255349500713266),
               ('Raw Pitch Accuracy', 0.9851298190401259),
               ('Raw Chroma Accuracy', 0.9853658536585366),
               ('Overall Accuracy', 0.7301076725130359)]),
                'AClassicEducation_NightOwl_STEM_13': OrderedDict([('Voicing Recall',
                0.995873786407767),
               ('Voicing False Alarm', 0.8500986193293886),
               ('Raw Pitch Accuracy', 0.9854368932038835), ....}

    """
    # YOUR CODE HERE

    # Loop through the tracks and assign the scores to the genre
    genre_scores = {}

    for track_id, score in scores_dictionary.items():
        if hasattr(tracks_dictionary[track_id], "genre"):
            genre = tracks_dictionary[track_id].genre
            if genre in genre_scores:
                genre_scores[genre][track_id] = score
            else:
                genre_scores[genre] = {track_id: score}

    # Return the dictionary with the scores by genre
    return genre_scores