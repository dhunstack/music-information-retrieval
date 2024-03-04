import numpy as np
from utils import estimate_beats_spectral_flux

def load_reference_beats(path):
    with open(path, 'r') as f:
        return np.array([float(line.strip()) for line in f])

def test_spectral_flux_output():
    audio_path = 'tests/data/gtzan_genre/gtzan_mini-main/genres/metal/metal.00002.wav'
    beats, novelty = estimate_beats_spectral_flux(audio_path)
    
    assert isinstance(beats, np.ndarray), "Expected beats to be a numpy array"
    assert isinstance(novelty, np.ndarray), "Expected novelty to be a numpy array"
    assert len(beats.shape) == 1, "Expected beats to be a 1-dimensional array"
    assert len(novelty.shape) == 1, "Expected novelty to be a 1-dimensional array"

def test_spectral_flux_beat_values():
    audio_path = 'tests/data/gtzan_genre/gtzan_mini-main/genres/metal/metal.00002.wav'
    reference_beats_path = 'tests/data/computed_beats_specflux.npy'
    computed_beats, _ = estimate_beats_spectral_flux(audio_path)
    reference_beats = np.load('tests/data/computed_beats_specflux.npy')

    assert np.allclose(computed_beats, reference_beats, atol=0.05), "Computed beats are not close to reference beats"

