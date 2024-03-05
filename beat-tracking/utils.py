import mirdata
import librosa
import mir_eval
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd

def load_data(dataset_name, data_home, dataset_version='1.0'):
    """
    Load a specific version of a dataset using the mirdata library.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset to load, e.g., "gtzan_genre".
    
    dataset_version : str
        Version of the dataset to load. To load the "mini" version, specify "mini". Default is "1.0".
    
    data_home : str
        Path to where the dataset is located. 

    Returns
    -------
    dataset : mirdata.Dataset
        The initialized mirdata Dataset object corresponding to the specified dataset and version.

    Notes
    -----
    The function is optimized for GTZAN-genre dataset but can be potentially used for other datasets supported by mirdata.
    """
    data = mirdata.initialize(dataset_name, data_home=data_home, version=dataset_version)
    return data

def estimate_beats_spectral_flux(audio_path, hop_length=512):
    """
    Compute beat positions using the spectral flux onset novelty function, followed by computing a tempogram and PLP.
    
    Parameters
    ----------
    audio_path : str
        Path to input audio file
    hop_length : int, optional
        Hop length, by default 512

    Returns
    -------
    beat_times : 1-d np.array
        Array of time stamps of the estimated beats in seconds.
    activation : 1-d np.array
        Array with the activation (or novelty function) values.
    """
    y, sr = librosa.load(audio_path)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    pulse = librosa.beat.plp(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
    
    # Calculate plp maximas for beats
    beats_plp = np.flatnonzero(librosa.util.localmax(pulse))
    times = librosa.times_like(onset_env, sr=sr)
    
    # Return values
    return times[beats_plp], pulse


def evaluate_estimated_beats(data, estimated_beats):
    """
    Evaluates the estimated beats for all tracks in the given data.

    Parameters:
        data (dict): Dictionary of tracks with track_id as the key and track information as the value.
        estimated_beats (dict): Dictionary with track_id as the key and array of estimated beat times as the value.

    Returns:
        dict: A dictionary with track_id as the key and evaluation score as the value.
    """
    evaluation_beats = {}

    for track_id, track in data.items():
        f_measure = mir_eval.beat.f_measure(track.beats.times, estimated_beats[track_id])
        evaluation_beats[track_id] = f_measure

    return evaluation_beats


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

    """
    genre_scores = {}

    for track_id, track in tracks_dictionary.items():
        if track.genre in genre_scores:
            genre_scores[track.genre][track_id] = scores_dictionary[track_id]
        else:
            genre_scores[track.genre] = {track_id: scores_dictionary[track_id]}

    return genre_scores

def get_tempo_vs_performance(scores_dictionary, tracks_dictionary):
    """Get score values as a function of tempo.

    Parameters
    ----------
    scores_dictionary : dict
        Dictionary of scores keyed by track_id
    tracks_dictionary: dict
        Dictionary of mirdata tracks keyed by track_id        

    Returns
    -------
    tempo : np.array
        Array of tempo values with the same number of elements as scores_dictionary
    scores : np.array
        Array of scores with the same number of elements as scores_dictionary
    """
    tempo = []; scores = []

    for track_id, track in tracks_dictionary.items():
        tempo.append(track.tempo)
        scores.append(scores_dictionary[track_id])

    return np.array(tempo), np.array(scores)


def compute_time_sf(novelty_sf, hop_size=512, sr=22050):
    """
    Compute the time axis for spectral flux novelty.

    Parameters
    ----------
    novelty_sf : np.array
        The spectral flux novelty curve.
    hop_size : int, optional
        Hop size used to compute the spectral flux. Default is 512.
    sr : int, optional
        Sampling rate of the signal. Default is 22050.

    Returns
    -------
    time_sf : np.array
        Time axis corresponding to the spectral flux novelty curve.
    """
    return np.arange(len(novelty_sf))*hop_size/sr


def compute_time_ml(novelty_ml, sr_ml=100):
    """
    Compute the time axis for machine learning activation function novelty.

    Parameters
    ----------
    novelty_ml : np.array
        The machine learning activation function novelty curve.
    sr_ml : int, optional
        Rate at which the novelty is computed (usually at 100Hz for madmom). Default is 100.

    Returns
    -------
    time_ml : np.array
        Time axis corresponding to the machine learning novelty curve.
    """
    return np.arange(len(novelty_ml))/sr_ml
    

def sonify_track_data(track_id, estimated_beats, tracks_dictionary):
    """
    Sonify the estimated beats for a given track ID.

    The purpose of this function is to sonify or "audify" the beats estimated 
    using different methods for a given track in the GTZAN dataset. This function 
    generates three audio clips for each track:
    1. The original audio with superimposed click sounds at the estimated beats.
    2. The original audio with superimposed click sounds at the reference beats.

    Steps:
    - Load the audio data for the provided track ID.
    - Select the estimated beats from the corresponding dictionary.
    - Generate click tracks (sonifications) for the estimated and reference beats. Hint: use mir_eval.sonify
    - Superimpose or add these click tracks to the original audio. Hint: make sure lengths of audio signals match.
    - Display the resulting audio clips for playback using IPython.display.

    Parameters
    ----------
    track_id : str
        A string representing the unique identifier for a track in the GTZAN 
        dataset.
    estimated_beats: dict
        Dictionary of estimated beats per track keyed by track_id
    tracks_dictionary: dict
        Dictionary of mirdata tracks keyed by track_id

    Returns
    -------
    None: 
        While the function doesn't return any values, it displays the audio clips 
        for playback in the environment (e.g., Jupyter notebook).
    """
    # Load audio and beats
    y, sr = tracks_dictionary[track_id].audio
    ref = tracks_dictionary[track_id].beats.times
    est = estimated_beats[track_id]
    
    # Generate click tracks for estimated and reference beats
    sonic_ref = mir_eval.sonify.clicks(ref, sr, length = len(y))
    sonic_est = mir_eval.sonify.clicks(est, sr, length = len(y))
    
    # Superimpose these click tracks to the original audio
    track_ref = y + sonic_ref
    track_est = y + sonic_est
    
    # Display the tracks
    print("Original Track")
    ipd.display(ipd.Audio(y, rate=sr))
    
    print("Track with Reference Beats")
    ipd.display(ipd.Audio(track_ref, rate=sr))
    
    print("Track with Estimated Beats")
    ipd.display(ipd.Audio(track_est, rate=sr))
    




