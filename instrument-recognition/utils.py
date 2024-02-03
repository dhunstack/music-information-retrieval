import mirdata
import librosa
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score 
from sklearn.neighbors import KNeighborsClassifier


def load_data(data_home):
    """
    Load the mini-Medley-Solos-DB dataset.

    Parameters
    ----------
    data_home : str
                Path to where the dataset is located

    Returns
    -------
    dataset: mirdata.Dataset
             The mirdata Dataset object correspondong to Medley-Solos-DB
    """
    
    # YOUR CODE HERE
    # Hints: 
    # Look at the mirdata tutorial on how to initialize a dataset.
    # Define the correct path using the data_home argument.
    data = mirdata.initialize('medley_solos_db', data_home=data_home)
    return data

def split_data(tracks):
    """
    Splits the provided dataset into training, validation, and test subsets based on the 'subset' 
    attribute of each track.

    Parameters
    ----------
    track_list : list
                 list of dataset.track objects from Medley_solos_DB dataset

    Returns
    -------
    tracks_train : list
        List of tracks belonging to the 'training' subset.
    tracks_validate : list
        List of tracks belonging to the 'validation' subset.
    tracks_test : list
        List of tracks belonging to the 'test' subset.
    """
    # YOUR CODE HERE
    tracks_train = []; tracks_validate = []; tracks_test = []
    
    for key, track in tracks.items():
        if track.subset == "training":
            tracks_train.append(track)
        elif track.subset == "test":
            tracks_test.append(track)
        elif track.subset == "validation":
            tracks_validate.append(track)
            
    return tracks_train, tracks_validate, tracks_test


def compute_mfccs(y, sr, n_fft=2048, hop_length=512, n_mels=128, n_mfcc=20):
    """
    Compute mfccs for an audio file using librosa, removing the 0th MFCC coefficient.
    
    Parameters
    ----------
    y : np.array
        Mono audio signal
    sr : int
        Audio sample rate
    n_fft : int
        Number of points for computing the fft
    hop_length : int
        Number of samples to advance between frames
    n_mels : int
        Number of mel frequency bands to use
    n_mfcc : int
        Number of mfcc's to compute
    
    Returns
    -------
    mfccs: np.array (t, n_mfcc - 1)
        Matrix of mfccs

    """
    # YOUR CODE HERE
    return librosa.feature.mfcc(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, n_mfcc=n_mfcc)[1:].T


def get_stats(features):
    """
    Compute summary statistics (mean and standard deviation) over a matrix of MFCCs.
    Make sure the statitics are computed across time (i.e. over all examples, 
    compute the mean of each feature).

    Parameters
    ----------
    features: np.array (n_examples, n_features)
              Matrix of features

    Returns
    -------
    features_mean: np.array (n_features)
                   The mean of the features
    features_std: np.array (n_features)
                   The standard deviation of the features

    """
    # Hint: use numpy mean and std functions, and watch out for the axis.
    # YOUR CODE HERE
    return np.mean(features, axis=0), np.std(features, axis=0)

def normalize(features, features_mean, features_std):
    """
    Normalize (standardize) a set of features using the given mean and standard deviation.

    Parameters
    ----------
    features: np.array (n_examples, n_features)
              Matrix of features
    features_mean: np.array (n_features)
              The mean of the features
    features_std: np.array (n_features)
              The standard deviation of the features

    Returns
    -------
    features_norm: np.array (n_examples, n_features)
                   Standardized features

    """

    # YOUR CODE HERE
    return (features-features_mean)/features_std


def get_features_and_labels(track_list, n_fft=2048, hop_length=512, n_mels=128, n_mfcc=20):
    """
    Our features are going to be the `mean` and `std` MFCC values of a track concatenated 
    into a single vector of size `2*n_mfcss`. 

    Create a function `get_features_and_labels()` such that extracts the features 
    and labels for all tracks in the dataset, such that for each audio file it obtains a 
    single feature vector. This function should do the following:

    For each track in the collection (e.g. training split),
        1. Compute the MFCCs of the input audio, and remove the first (0th) coeficient.
        2. Compute the summary statistics of the MFCCs over time:
            1. Find the mean and standard deviation for each MFCC feature (2 values for each)
            2. Stack these statistics into single 1-d vector of lenght ( 2 * (n_mfccs - 1) )
        3. Get the labels. The label of a track can be accessed by calling `track.instrument_id`.
    Return the labels and features as `np.arrays`.

    Parameters
    ----------
    track_list : list
                 list of dataset.track objects from Medley_solos_DB dataset
    n_fft : int
                 Number of points for computing the fft
    hop_length : int
                 Number of samples to advance between frames
    n_mels : int
             Number of mel frequency bands to use
    n_mfcc : int
             Number of mfcc's to compute

    Returns
    -------
    feature_matrix: np.array (len(track_list), 2*(n_mfcc - 1))
        The features for each track, stacked into a matrix.
    label_array: np.array (len(track_list))
        The label for each track, represented as integers
    """

    # Hint: re-use functions from previous parts (e.g. compute_mfcss and get_stats)
    # YOUR CODE HERE
    feature_matrix = np.zeros(2 * (n_mfcc - 1))
    label_array = []
    
    for track in track_list:
        y, sr = track.audio
        mfcc = compute_mfccs(y, sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, n_mfcc=n_mfcc)
        mean, std = get_stats(mfcc)
        feature_matrix = np.vstack(( feature_matrix, np.hstack((mean, std)) ))
        label_array.append(track.instrument_id)
        
    return feature_matrix[1:], np.array(label_array)

def fit_knn(train_features, train_labels, validation_features, validation_labels, ks=[1, 5, 10, 50]):
    """
    Fit a k-nearest neighbor classifier and choose the k which maximizes the
    *f-measure* on the validation set.
    
    Plot the f-measure on the validation set as a function of k.

    Parameters
    ----------
    train_features : np.array (n_train_examples, n_features)
        training feature matrix
    train_labels : np.array (n_train_examples)
        training label array
    validation_features : np.array (n_validation_examples, n_features)
        validation feature matrix
    validation_labels : np.array (n_validation_examples)
        validation label array
    ks: list of int
        k values to evaluate using the validation set

    Returns
    -------
    knn_clf : scikit learn classifier
        Trained k-nearest neighbor classifier with the best k
    best_k : int
        The k which gave the best performance
    """
    
    # Hint: for simplicity you can search over k = 1, 5, 10, 50. 
    # Use KNeighborsClassifier from sklearn.
    # YOUR CODE HERE
    max_f1 = -1; best_k = 0; knn_clf = 0; all_f1 = []
    for k in ks:
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(train_features, train_labels)
        validation_pred = clf.predict(validation_features)
        f1 = f1_score(validation_labels, validation_pred, average='weighted')
        all_f1.append(f1)
        
        if f1>max_f1:
            max_f1 = f1
            best_k = k
            knn_clf = clf
    
    plt.xlabel("k")
    plt.ylabel("f1 score")
    plt.plot(ks, all_f1, '-x')
    return knn_clf, best_k
    
    
        