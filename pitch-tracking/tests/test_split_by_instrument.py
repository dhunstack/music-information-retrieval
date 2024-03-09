from unittest.mock import Mock
from utils import split_by_instrument  # Assuming the function is in 'utils.py'


class MockTrack:
    """Mock track object with a genre and tempo attribute."""
    def __init__(self, instrument=None):
        self.instrument = instrument

def test_split_by_instrument():


    # Mock scores
    scores_dict = {
        'track1': {'score_1': 0.9, 'score_2': 0.9},
        'track2': {'score_1': 0.85, 'score_2': 0.85},
        'track3': {'score_1': 0.92, 'score_2': 0.92},
        'track4': {'score_1': 0.88, 'score_2': 0.88},
    }

    # Mock dataset using MockTrack objects with genres
    mock_tracks = {
        'track1': MockTrack(instrument='guitar'),
        'track2': MockTrack(instrument='piano'),
        'track3': MockTrack(instrument='drums'),
        'track4': MockTrack(instrument='bass'),
    }

    expected_instruments = ['guitar', 'piano', 'drums', 'bass']
    results = split_by_instrument(scores_dict, mock_tracks)
    print(results)

    # Check if results contain only the top 6 instruments
    assert set(results.keys()) == set(expected_instruments), "Unexpected instruments in result"

    # Check if scores are preserved for each track
    for instrument in results:
        for track_id, score in results[instrument].items():
            assert score == scores_dict[track_id], f"Mismatch in scores for {track_id}"

