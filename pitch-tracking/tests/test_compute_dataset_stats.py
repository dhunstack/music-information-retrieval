from unittest.mock import Mock, patch
from utils import compute_dataset_stats  

def test_compute_dataset_stats():
    # Mocking a dataset with 3 tracks
    mock_dataset = Mock()
    mock_dataset.track_ids = ['track1', 'track2', 'track3']
    
    # Mocking tracks
    track1 = Mock()
    track1.instrument = 'guitar'
    track1.genre = 'rock'
    
    track2 = Mock()
    track2.instrument = 'piano'
    track2.genre = 'jazz'
    
    track3 = Mock()
    track3.instrument = 'guitar'
    track3.genre = 'rock'
    
    # Mapping track IDs to mock tracks
    track_id_to_mock_track = {
        'track1': track1,
        'track2': track2,
        'track3': track3,
    }

    # When dataset.track(track_id) is called, return the appropriate mock track
    mock_dataset.track.side_effect = lambda track_id: track_id_to_mock_track[track_id]

    # Testing the compute_dataset_stats function
    stats = compute_dataset_stats(mock_dataset)

    assert stats['num_tracks'] == 3, f"Expected 3, got {stats['num_tracks']}"
    assert stats['instrument_distribution'] == {'guitar': 2, 'piano': 1}, f"Expected {'guitar': 2, 'piano': 1}, got {stats['instrument_distribution']}"
    assert stats['genre_distribution'] == {'rock': 2, 'jazz': 1}, f"Expected {'rock': 2, 'jazz': 1}, got {stats['genre_distribution']}"

