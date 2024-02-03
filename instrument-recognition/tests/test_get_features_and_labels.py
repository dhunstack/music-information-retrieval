import unittest
from unittest.mock import patch, Mock
import numpy as np

from utils import get_features_and_labels

class MockTrack:
    def __init__(self, instrument_id, audio):
        self.instrument_id = instrument_id
        self.audio = audio

def test_get_features_and_labels():
    # Dummy audio signals and sample rates for our mock tracks
    dummy_audios = [
        (np.array([0.1, 0.2, 0.3]), 44100),
        (np.array([0.4, 0.5, 0.6]), 44100)
    ]
    
    # Mock tracks
    mock_tracks = [MockTrack(0, dummy_audios[0]), MockTrack(1, dummy_audios[1])]

    # Mocked return values for compute_mfccs and get_stats
    mocked_mfcc = np.ones((3, 19))  # 19 as 20-1
    mocked_mean = np.ones(19)
    mocked_std = np.ones(19)

    with patch('utils.compute_mfccs', return_value=mocked_mfcc) as mock_compute_mfccs:
        with patch('utils.get_stats', return_value=(mocked_mean, mocked_std)) as mock_get_stats:
            
            features, labels = get_features_and_labels(mock_tracks)
            
            # Check shape of the returned features and labels
            assert features.shape == (2, 38), "Feature matrix shape is incorrect"  # 2*(20-1)
            assert labels.shape == (2,), "Labels array shape is incorrect"

            # Check values in the feature matrix (mean and std are all ones)
            assert np.array_equal(features, np.ones((2, 38))), "Feature values are incorrect."
            
            # Check the label values
            assert np.array_equal(labels, np.array([0, 1])), "Label values are incorrect."

            # Assert the mock functions were called the right number of times
            assert mock_compute_mfccs.call_count == 2, "compute_mfccs not called twice!"
            assert mock_get_stats.call_count == 2, "get_stats not called twice!"
