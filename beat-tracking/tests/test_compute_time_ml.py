import numpy as np
from utils import compute_time_ml

def test_compute_time_ml():
    # Mock data
    novelty_ml = np.array([0.1, 0.2, 0.3, 0.4])
    
    # Default sr_ml value
    sr_ml = 100
    
    # Expected result
    expected_time_ml = np.load('tests/data/expected_time_ml1.npy')
    
    # Test function with default sr_ml
    computed_time_ml = compute_time_ml(novelty_ml)
    assert np.array_equal(computed_time_ml, expected_time_ml), "Mismatch in computed time axis with default sr_ml"

    # Additional test with different sr_ml value
    sr_ml_2 = 200
    expected_time_ml_2 = np.load('tests/data/expected_time_ml2.npy')
    computed_time_ml_2 = compute_time_ml(novelty_ml, sr_ml_2)
    assert np.array_equal(computed_time_ml_2, expected_time_ml_2), "Mismatch in computed time axis with different sr_ml"

