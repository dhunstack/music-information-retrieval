import numpy as np
from utils import compute_time_sf

def test_compute_time_sf():
    # Mock data
    novelty_sf = np.array([0.1, 0.2, 0.3, 0.4])
    hop_size = 512
    sr = 22050
    
    # Expected result
    expected_time_sf = np.load('tests/data/expected_time_sf1.npy')
    
    # Test function
    computed_time_sf = compute_time_sf(novelty_sf, hop_size, sr)
    assert np.array_equal(computed_time_sf, expected_time_sf), "Mismatch in computed time axis"

    # Additional test with different hop_size and sr values
    hop_size_2 = 256
    sr_2 = 44100
    expected_time_sf_2 = np.load('tests/data/expected_time_sf2.npy')
    computed_time_sf_2 = compute_time_sf(novelty_sf, hop_size_2, sr_2)
    assert np.array_equal(computed_time_sf_2, expected_time_sf_2), "Mismatch in computed time axis with different hop_size and sr"
