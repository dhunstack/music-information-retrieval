import mir_eval
import numpy as np
from utils import evaluate_estimated_beats

# Mock function
def mock_f_measure(reference_beats, estimated_beats):
    # Mock a simple f_measure calculation for the purpose of this test
    # (This isn't the real formula but is okay for the mock)
    return len(set(reference_beats).intersection(set(estimated_beats))) / len(reference_beats)

class MockTrack:
    """Mock track object with a genre and tempo attribute."""
    def __init__(self, audio_path=None, beats=None, tempo=None):
        self.audio_path = audio_path
        self.beats = type('', (), {})()
        self.beats.times = np.array([0.5, 1.0, 1.5])
        self.tempo = 120

# Apply Mock
mir_eval.beat.f_measure = mock_f_measure

def test_evaluate_estimated_beats():
    # Mock Data
    mock_data = {
        'track1': MockTrack(audio_path = 'mock_audio_path1'),
        'track2': MockTrack(audio_path = 'mock_audio_path2'),
    }
    mock_estimated_beats = {
        'track1': [0.5, 1.0, 1.5],
        'track2': [0.5, 1.1, 1.5]
    }
    
    scores = evaluate_estimated_beats(mock_data, mock_estimated_beats)
    assert scores['track1'] == 1.0, "Expected F-measure of 1.0 for track1"
    assert scores['track2'] == 2/3, "Expected F-measure of 2/3 for track2"
