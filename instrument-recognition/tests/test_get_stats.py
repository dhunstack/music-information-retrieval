import mirdata
import numpy as np
from utils import get_stats

def test_get_stats():
    # Create a dummy features matrix
    features = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])

    # Compute the statistics using the function
    features_mean, features_std = get_stats(features)

    # Compute the statistics directly using numpy for comparison
    expected_mean = np.mean(features, axis=0)
    expected_std = np.std(features, axis=0)

    # Check that the mean and std are computed correctly
    assert np.array_equal(features_mean, expected_mean), "Computed mean does not match expected values. Are you using the right axis?"
    assert np.array_equal(features_std, expected_std), "Computed standard deviation does not match expected values. Are you using the right axis?"

    # Check that the output shapes are correct
    assert features_mean.shape == (3,), "Mean shape is incorrect."
    assert features_std.shape == (3,), "Standard deviation shape is incorrect."

