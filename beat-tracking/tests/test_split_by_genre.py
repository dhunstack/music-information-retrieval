from utils import split_by_genre

class MockTrack:
    """Mock track object with a genre and tempo attribute."""
    def __init__(self, genre=None, tempo=None):
        self.genre = genre

def test_split_by_genre():
    # Mock scores
    scores_dict = {
        'track1': 0.9,
        'track2': 0.85,
        'track3': 0.92,
        'track4': 0.88,
    }

    # Mock dataset using MockTrack objects with genres
    mock_tracks = {
        'track1': MockTrack(genre='rock'),
        'track2': MockTrack(genre='pop'),
        'track3': MockTrack(genre='rock'),
        'track4': MockTrack(genre='jazz'),
    }

    genre_scores = split_by_genre(scores_dict, mock_tracks)

    # Test if function returns a dictionary with genre as keys
    assert set(genre_scores.keys()) == {'rock', 'pop', 'jazz'}, "Expected genres not found"

    # Test if tracks are correctly categorized by their genres
    assert 'track1' in genre_scores['rock'], "Expected track1 in rock genre"
    assert 'track2' in genre_scores['pop'], "Expected track2 in pop genre"
    assert 'track3' in genre_scores['rock'], "Expected track3 in rock genre"
    assert 'track4' in genre_scores['jazz'], "Expected track4 in jazz genre"

    # Test if scores are correctly mapped
    assert genre_scores['rock']['track1'] == 0.9, "Incorrect score for track1"
    assert genre_scores['pop']['track2'] == 0.85, "Incorrect score for track2"
    assert genre_scores['rock']['track3'] == 0.92, "Incorrect score for track3"
    assert genre_scores['jazz']['track4'] == 0.88, "Incorrect score for track4"

