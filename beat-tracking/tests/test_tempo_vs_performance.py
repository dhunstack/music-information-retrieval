import numpy as np
from utils import get_tempo_vs_performance


class MockTrack:
    """Mock track object with a tempo attribute."""
    def __init__(self, tempo):
        self.tempo = tempo

def test_get_tempo_vs_score():
    # Mock scores
    scores_dict = {
        'track1': 0.9,
        'track2': 0.85,
        'track3': 0.92,
        'track4': 0.88,
    }

    # Mock dataset using MockTrack objects
    mock_tracks = {
        'track1': MockTrack(120),
        'track2': MockTrack(130),
        'track3': MockTrack(140),
        'track4': MockTrack(150),
    }
    
    tempos, scores = get_tempo_vs_performance(scores_dict, mock_tracks)

    # Check if outputs are numpy arrays
    assert isinstance(tempos, np.ndarray), "Expected tempos to be a numpy array"
    assert isinstance(scores, np.ndarray), "Expected scores to be a numpy array"

    # Check if lengths are correct
    assert len(tempos) == len(scores_dict), "Mismatched length for tempos"
    assert len(scores) == len(scores_dict), "Mismatched length for scores"

    # Check if values are correctly returned
    expected_tempos = np.array([120, 130, 140, 150])
    expected_scores = np.array([0.9, 0.85, 0.92, 0.88])
    
    assert np.all(tempos == expected_tempos), "Tempos do not match expected values"
    assert np.all(scores == expected_scores), "Scores do not match expected values"


