import madmom
import librosa
import numpy as np
import matplotlib.pyplot as plt
from utils import compute_time_sf, compute_time_ml, estimate_beats_spectral_flux

# ================== AUXILIARY FUNCTIONS ===========================

# DO NOT MODIFY THIS FUNCTIONS, THEY ARE FOR AUXILIARY USE

def estimate_beats_madmom(audio_path):
    """
    Compute beat positions using a machine learned onset novelty function with madmom.
    
    Parameters
    ----------
    audio_path : str
        Path to input audio file

    Returns
    -------
    beat_times : 1-d np.array
        Array of time stamps of the estimated beats in seconds.
    activation : 1-d np.array
        Array with the activation (or novelty function) values.
    """

    # load the audio (make sure sampling rate is compatible with madmom)
    y, sr = librosa.load(audio_path, sr=44100)
    # use RNNBeatProcessor() from madmom to obtain an activation function
    proc = madmom.features.beats.RNNBeatProcessor()
    act = proc(y)
    # use DBNBeatTrackingProcessor from madmom to obtain the beat time stamps
    dbn = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
    beats = dbn(act)
    novelty = act

    return beats, novelty

def get_estimated_beats(data, method):
    """
    Computes the estimated beats for all tracks in the given data using the specified method.

    Parameters:
        data (dict): Dictionary of tracks with track_id as the key and track information as the value.
        method (str): Either "spectral_flux" or "machine_learning".

    Returns:
        dict: A dictionary with track_id as the key and estimated beat times as the value.
    """
    estimated_beats = {}

    for track_id, track in data.items():
        if method == "spectral_flux":
            est_beat_times, _ = estimate_beats_spectral_flux(track.audio_path)
        elif method == "machine_learning":
            est_beat_times, _ = estimate_beats_madmom(track.audio_path)
        else:
            raise ValueError("Invalid method. Choose either 'spectral_flux' or 'machine_learning'.")
        
        estimated_beats[track_id] = est_beat_times

    return estimated_beats


def compute_track_data(track_id, tracks_dictionary, hop_size=512):
    """
    Compute the necessary data for plotting beat estimations, novelty functions and ground truth given a track ID.

    Parameters
    ----------
    track_id : str
        GTZAN track_id
    tracks_dictionary: dict
        Dictionary of mirdata tracks keyed by track_id

    Returns
    -------
    data_dict : dict
        Dictionary containing the necessary arrays and values for plotting. 

        'novelty_sf': np.array
            The novelty curve computed using spectral flux, computed with a hopsize of 512.
        
        'time_sf': np.array
            The temporal axis corresponding to 'novelty_sf'.
        
        'beat_times_sf': np.array
            Estimated beat times using spectral flux novelty curve.
        
        'novelty_ml': np.array
            The novelty curve computed using a machine learning activation function.
        
        'time_ml': np.array
            The temporal axis corresponding to 'novelty_ml'.
        
        'beat_times_ml': np.array
            Estimated beat times using machine learning activation function novelty curve.
        
        'audio': np.array
            The audio waveform of the track.
        
        'time': np.array
            The temporal axis corresponding to 'audio'.
        
        'beats': np.array
            Reference beat times (ground truth).
    """
    # Hint: to estimate the time for the spectral flux novelty, take into account the sampling rate of the signal and the hop-size 
    # used to compute the spectral flux. For the madmom novelty, take into account that the novelty in madmom is computed at 100Hz.

    track = tracks_dictionary[track_id]
    audio_path = track.audio_path

    # Compute novelties and beat times for spectral flux
    beat_times_sf, novelty_sf = estimate_beats_spectral_flux(audio_path, hop_length=hop_size)
    time_sf = compute_time_sf(novelty_sf, hop_size)
    
    # Compute novelties and beat times for machine learning activation function
    beat_times_ml, novelty_ml = estimate_beats_madmom(audio_path)
    time_ml = compute_time_ml(novelty_ml)
    
    # Load audio and compute time
    audio, sr = librosa.load(audio_path, sr=44100)
    time = librosa.samples_to_time(np.arange(len(audio)), sr=44100)
    beats = track.beats.times

    return {
        'novelty_sf': novelty_sf,
        'time_sf': time_sf,
        'beat_times_sf': beat_times_sf,
        'novelty_ml': novelty_ml,
        'time_ml': time_ml,
        'beat_times_ml': beat_times_ml,
        'audio': audio,
        'time': time,
        'beats': beats
    }


def plot_track_data(track_data):
    """
    Plot the track data obtained from the `compute_track_data` function.

    Parameters
    ----------
    track_data : dict
        Dictionary containing the necessary arrays and values for plotting. 

    Returns
    -------
    None
    """
    
    fig, ax = plt.subplots(3, 1, figsize=(10, 5))

    # Spectral flux novelty curve + the estimated beats
    ax[0].plot(track_data['time_sf'], track_data['novelty_sf'], label='Spectral flux', linewidth=1.5)
    ax[0].vlines(track_data['beat_times_sf'], 0, 1, label='Beats SF', color='green', linewidths=1.2)
    ax[0].set_title("Spectral Flux Novelty Curve and Estimated Beats")
    ax[0].set_ylabel('Amplitude')
    ax[0].legend(frameon=True, framealpha=1.0, edgecolor='black')

    # Machine learned activation function + the estimated beats
    ax[1].plot(track_data['time_ml'], track_data['novelty_ml'], label='ML activation', linewidth=1.5, color='blue')
    ax[1].vlines(track_data['beat_times_ml'], 0, 1, label='Beats ML', color='red', linewidths=1.2)
    ax[1].set_title("Machine Learning Activation Function and Estimated Beats")
    ax[1].set_ylabel('Amplitude')
    ax[1].legend(frameon=True, framealpha=1.0, edgecolor='black')

    # Audio waveform + reference beats
    ax[2].plot(track_data['time'], track_data['audio'], label="audio waveform", linewidth=0.7, color='grey')
    ax[2].vlines(track_data['beats'], min(track_data['audio']), max(track_data['audio']), label="Reference beats", linewidths=0.9, colors='purple')
    ax[2].set_title("Audio Waveform and Reference Beats")
    ax[2].set_xlabel('Time (s)')
    ax[2].set_ylabel('Amplitude')
    ax[2].legend(frameon=True, framealpha=1.0, edgecolor='black')

    # Adjust layout for clarity
    plt.tight_layout()
    plt.show()


def plot_overall_scores(overall_scores):
    """
    Plot a boxplot visualization of the F-measure scores for two methods: Spectral Flux and Machine Learning.

    This function takes in the overall F-measure scores for two methods and plots them as boxplots 
    side by side. The plots are styled for clarity, and each boxplot is colored to differentiate 
    between the two methods. Various boxplot components are customized for improved aesthetics.

    Parameters
    ----------
    overall_scores : list of lists
        A list containing two lists. The first list should contain the F-measure scores for tracks 
        evaluated using the Spectral Flux method, and the second list should contain the scores for 
        tracks evaluated using the Machine Learning method. 
        For instance, overall_scores should look something like this:
        [[score1, score2, ...], [score1, score2, ...]]

    Returns
    -------
    None : The function will directly display the plot.

    Examples
    --------
    >>> scores_sf = [0.9, 0.85, 0.88, ...]
    >>> scores_ml = [0.92, 0.83, 0.89, ...]
    >>> plot_overall_scores([scores_sf, scores_ml])

    """

    # Set style
    # plt.style.use('seaborn-whitegrid')
    plt.style.use('seaborn-v0_8-whitegrid')

    fig, ax = plt.subplots()

    # Customize the boxplot appearance
    boxprops = dict(linewidth=2, color='darkblue')
    medianprops = dict(linewidth=2, color='darkred')
    whiskerprops = dict(linewidth=2, color='darkblue')
    capprops = dict(linewidth=2, color='darkblue')

    ax.boxplot(overall_scores, boxprops=boxprops, medianprops=medianprops,
               whiskerprops=whiskerprops, capprops=capprops, patch_artist=True)

    # Customize the background color
    for box in ax.patches:
        box.set_facecolor('lightblue')

    # Title and labels
    ax.set_title('F-measure for different methods', fontsize=15, fontweight='bold')
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['spectral flux', 'machine learning'], fontsize=13)
    ax.set_ylabel('F-measure', fontsize=13)

    # Tick label size
    ax.tick_params(axis='both', which='major', labelsize=12)

    plt.show()

def plot_scores_by_genre(genre_scores_sf, genre_scores_ml):
    """
    Visualizes the F-measure scores by genre for two methods: Spectral Flux and Machine Learning.

    The function creates a boxplot for each genre. The left plot represents scores for tracks evaluated 
    using the Spectral Flux method, and the right plot represents scores for tracks evaluated using the 
    Machine Learning method. The x-axis corresponds to the genre, and the y-axis corresponds to the 
    F-measure scores.

    Each box plot is colored to differentiate between the two methods, and various boxplot components 
    are customized for improved aesthetics.
    
    Parameters
    ----------
    genre_scores_sf : dict
        Dictionary containing scores categorized by genre for the Spectral Flux method.
        
    genre_scores_ml : dict
        Dictionary containing scores categorized by genre for the Machine Learning method.

    Returns
    -------
    None : The function will directly display the plot.
    """

    # Set style
    # plt.style.use('seaborn-whitegrid')
    plt.style.use('seaborn-v0_8-whitegrid')

    # Data and genres
    genres = [genre for genre, d in genre_scores_ml.items()]

    # Customize the boxplot appearance
    boxprops = dict(linewidth=1.5, color='darkblue')
    medianprops = dict(linewidth=1.5, color='darkred')
    whiskerprops = dict(linewidth=1.5, color='darkblue')
    capprops = dict(linewidth=1.5, color='darkblue')

    # Figure setup
    fig, ax = plt.subplots(1, 2, sharey=True, figsize=(10, 6))
    fig.subplots_adjust(wspace=0.1)

    # Left plot
    ax[0].boxplot([list(d.values()) for genre, d in genre_scores_sf.items()], 
                  boxprops=boxprops, medianprops=medianprops, whiskerprops=whiskerprops, 
                  capprops=capprops, patch_artist=True)
    ax[0].set_title('Spectral Flux', fontsize=15, fontweight='bold')
    ax[0].set_xticks(list(range(1, len(genres)+1)))
    ax[0].set_xticklabels(genres, rotation=45, fontsize=11)
    ax[0].set_ylabel('F-measure', fontsize=13)

    # Right plot
    ax[1].boxplot([list(d.values()) for genre, d in genre_scores_ml.items()], 
                  boxprops=boxprops, medianprops=medianprops, whiskerprops=whiskerprops, 
                  capprops=capprops, patch_artist=True)
    ax[1].set_title('Machine Learning', fontsize=15, fontweight='bold')
    ax[1].set_xticks(list(range(1, len(genres)+1)))
    ax[1].set_xticklabels(genres, rotation=45, fontsize=11)

    # Color the boxes
    for patch in ax[0].patches:
        patch.set_facecolor('lightblue')
    for patch in ax[1].patches:
        patch.set_facecolor('lightgreen')

    plt.show()



def plot_scores_by_tempo(tempo_sf, score_by_tempo_sf, tempo_ml, score_by_tempo_ml):
    """
    Visualizes the F-measure scores against tempo for two methods: Spectral Flux and Machine Learning.

    The function creates a scatter plot with two subplots - one for each method. The top subplot 
    represents the Spectral Flux method and the bottom subplot represents the Machine Learning method. 
    The shared x-axis corresponds to the tempo (in BPM), and the y-axis corresponds to the F-measure scores.
    
    Parameters
    ----------
    tempo_sf : np.array or list
        An array or list containing the tempos for the tracks evaluated using the Spectral Flux method.
        
    score_by_tempo_sf : np.array or list
        An array or list containing the F-measure scores corresponding to the tempos for the tracks 
        evaluated using the Spectral Flux method.
        
    tempo_ml : np.array or list
        An array or list containing the tempos for the tracks evaluated using the Machine Learning method.
        
    score_by_tempo_ml : np.array or list
        An array or list containing the F-measure scores corresponding to the tempos for the tracks 
        evaluated using the Machine Learning method.

    Returns
    -------
    None : The function will directly display the plot.
    """

    # Set style
    # plt.style.use('seaborn-whitegrid')
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Figure setup
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
    fig.subplots_adjust(hspace=0.4)

    # Top subplot
    ax[0].scatter(x=tempo_sf, y=score_by_tempo_sf, color='blue', alpha=0.7)
    ax[0].set_title('Spectral Flux', fontsize=15, fontweight='bold')
    ax[0].set_ylabel('F-measure', fontsize=13)

    # Bottom subplot
    ax[1].scatter(x=tempo_ml, y=score_by_tempo_ml, color='green', alpha=0.7)
    ax[1].set_title('Machine Learning', fontsize=15, fontweight='bold')
    ax[1].set_xlabel('Tempo (BPM)', fontsize=13)
    ax[1].set_ylabel('F-measure', fontsize=13)

    plt.show()
