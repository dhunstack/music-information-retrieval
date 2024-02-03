import numpy as np
import librosa
from utils import compute_mfccs

def test_compute_mfccs():
    # Load an audio signal
    y, sr = librosa.load('tests/data/medley_solos_db/audio/Medley-solos-DB_test-0_0ef1f0ea-e3ce-5997-fbdc-631376875bfe.wav')

    # Compute MFCCs using the provided function
    mfccs = compute_mfccs(y, sr, n_fft=1024)

    # Load expected mfccs
    mfcc_librosa = librosa.feature.mfcc(y=y, sr=sr, n_fft=1024).T[:, 1:]	

    # Assert that the shapes are correct
    assert mfccs.shape[1] == 19, "Number of MFCC coefficients is not n_mfcc - 1"
    assert mfccs.shape == mfcc_librosa.shape, "MFCC shape is not the expected one!"

    # Check that the MFCCs are close to the direct computation
    assert np.allclose(mfccs, mfcc_librosa, atol=1e-6), "Computed MFCCs do not match expected values"
    
    # Check if the 0th MFCC coefficient is removed
    assert not np.array_equal(mfccs, librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, n_fft=2048, hop_length=512, n_mels=128).T), \
        "0th MFCC coefficient not removed"
