import mirdata
import numpy as np
from utils import normalize

def test_normalize():
    # Create a dummy features matrix
    features = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    
    # Dummy mean and std arrays
    features_mean = np.array([4, 5, 6])
    features_std = np.array([2, 2, 2])

    # Normalize the features using the function
    features_norm = normalize(features, features_mean, features_std)

    # Compute the normalized features directly for comparison
    expected_features_norm = np.load('tests/data/features_normalized.npy')

    # Check that the normalized features match the expected values
    assert np.array_equal(features_norm, expected_features_norm), "Normalized features do not match expected values."

    # Check that the shape of normalized features is correct
    assert features_norm.shape == (3, 3), "Normalized features shape is incorrect."
