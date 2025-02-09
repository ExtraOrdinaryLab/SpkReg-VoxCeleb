import math
from packaging import version
from dataclasses import dataclass
from abc import ABC, abstractmethod

import torch

try:
    import torchaudio
    import torchaudio.functional
    import torchaudio.transforms

    TORCHAUDIO_VERSION = version.parse(torchaudio.__version__)
    TORCHAUDIO_VERSION_MIN = version.parse('0.5')

    HAVE_TORCHAUDIO = True
except ModuleNotFoundError:
    HAVE_TORCHAUDIO = False

from .logging import logger
from .module import NeuralModule
from .features import FilterbankFeatures, FilterbankFeaturesTA
from .spectrogram_augment import SpecCutout, SpecAugment


class AudioPreprocessor(NeuralModule, ABC):
    """
    An interface for Neural Modules that performs audio pre-processing,
    transforming the wav files to features.
    """

    def __init__(self, win_length, hop_length):
        super().__init__()

        self.win_length = win_length
        self.hop_length = hop_length

        self.torch_windows = {
            'hann': torch.hann_window,
            'hamming': torch.hamming_window,
            'blackman': torch.blackman_window,
            'bartlett': torch.bartlett_window,
            'ones': torch.ones,
            None: torch.ones,
        }

        # Normally, when you call to(dtype) on a torch.nn.Module, all
        # floating point parameters and buffers will change to that
        # dtype, rather than being float32. The AudioPreprocessor
        # classes, uniquely, don't actually have any parameters or
        # buffers from what I see. In addition, we want the input to
        # the preprocessor to be float32, but need to create the
        # output in appropriate precision. We have this empty tensor
        # here just to detect which dtype tensor this module should
        # output at the end of execution.
        self.register_buffer("dtype_sentinel_tensor", torch.tensor((), dtype=torch.float32), persistent=False)

    @torch.no_grad()
    def forward(self, input_signal, length):
        processed_signal, processed_length = self.get_features(input_signal.to(torch.float32), length)
        processed_signal = processed_signal.to(self.dtype_sentinel_tensor.dtype)
        return processed_signal, processed_length

    @abstractmethod
    def get_features(self, input_signal, length):
        # Called by forward(). Subclasses should implement this.
        pass


class AudioToMelSpectrogramPreprocessor(AudioPreprocessor):
    """Featurizer module that converts wavs to mel spectrograms.

    Args:
        sample_rate (int): Sample rate of the input audio data.
            Defaults to 16000
        window_size (float): Size of window for fft in seconds
            Defaults to 0.02
        window_stride (float): Stride of window for fft in seconds
            Defaults to 0.01
        n_window_size (int): Size of window for fft in samples
            Defaults to None. Use one of window_size or n_window_size.
        n_window_stride (int): Stride of window for fft in samples
            Defaults to None. Use one of window_stride or n_window_stride.
        window (str): Windowing function for fft. can be one of ['hann',
            'hamming', 'blackman', 'bartlett']
            Defaults to "hann"
        normalize (str): Can be one of ['per_feature', 'all_features']; all
            other options disable feature normalization. 'all_features'
            normalizes the entire spectrogram to be mean 0 with std 1.
            'pre_features' normalizes per channel / freq instead.
            Defaults to "per_feature"
        n_fft (int): Length of FT window. If None, it uses the smallest power
            of 2 that is larger than n_window_size.
            Defaults to None
        preemph (float): Amount of pre emphasis to add to audio. Can be
            disabled by passing None.
            Defaults to 0.97
        features (int): Number of mel spectrogram freq bins to output.
            Defaults to 64
        lowfreq (int): Lower bound on mel basis in Hz.
            Defaults to 0
        highfreq  (int): Lower bound on mel basis in Hz.
            Defaults to None
        log (bool): Log features.
            Defaults to True
        log_zero_guard_type(str): Need to avoid taking the log of zero. There
            are two options: "add" or "clamp".
            Defaults to "add".
        log_zero_guard_value(float, or str): Add or clamp requires the number
            to add with or clamp to. log_zero_guard_value can either be a float
            or "tiny" or "eps". torch.finfo is used if "tiny" or "eps" is
            passed.
            Defaults to 2**-24.
        dither (float): Amount of white-noise dithering.
            Defaults to 1e-5
        pad_to (int): Ensures that the output size of the time dimension is
            a multiple of pad_to.
            Defaults to 16
        frame_splicing (int): Defaults to 1
        exact_pad (bool): If True, sets stft center to False and adds padding, such that num_frames = audio_length
            // hop_length. Defaults to False.
        pad_value (float): The value that shorter mels are padded with.
            Defaults to 0
        mag_power (float): The power that the linear spectrogram is raised to
            prior to multiplication with mel basis.
            Defaults to 2 for a power spec
        rng : Random number generator
        nb_augmentation_prob (float) : Probability with which narrowband augmentation would be applied to
            samples in the batch.
            Defaults to 0.0
        nb_max_freq (int) : Frequency above which all frequencies will be masked for narrowband augmentation.
            Defaults to 4000
        use_torchaudio: Whether to use the `torchaudio` implementation.
        mel_norm: Normalization used for mel filterbank weights.
            Defaults to 'slaney' (area normalization)
        stft_exact_pad: Deprecated argument, kept for compatibility with older checkpoints.
        stft_conv: Deprecated argument, kept for compatibility with older checkpoints.
    """

    def __init__(
        self,
        sample_rate=16000,
        window_size=0.02,
        window_stride=0.01,
        n_window_size=None,
        n_window_stride=None,
        window="hann",
        normalize="per_feature",
        n_fft=None,
        preemph=0.97,
        features=64,
        lowfreq=0,
        highfreq=None,
        log=True,
        log_zero_guard_type="add",
        log_zero_guard_value=2**-24,
        dither=1e-5,
        pad_to=16,
        frame_splicing=1,
        exact_pad=False,
        pad_value=0,
        mag_power=2.0,
        rng=None,
        nb_augmentation_prob=0.0,
        nb_max_freq=4000,
        use_torchaudio: bool = False,
        mel_norm="slaney",
    ):
        super().__init__(n_window_size, n_window_stride)

        self._sample_rate = sample_rate
        if window_size and n_window_size:
            raise ValueError(f"{self} received both window_size and " f"n_window_size. Only one should be specified.")
        if window_stride and n_window_stride:
            raise ValueError(
                f"{self} received both window_stride and " f"n_window_stride. Only one should be specified."
            )
        if window_size:
            n_window_size = int(window_size * self._sample_rate)
        if window_stride:
            n_window_stride = int(window_stride * self._sample_rate)

        # Given the long and similar argument list, point to the class and instantiate it by reference
        if not use_torchaudio:
            logger.warning("Current only support FilterbankFeatures with torchaudio.")
            featurizer_class = FilterbankFeaturesTA
        else:
            featurizer_class = FilterbankFeaturesTA
        self.featurizer = featurizer_class(
            sample_rate=self._sample_rate,
            n_window_size=n_window_size,
            n_window_stride=n_window_stride,
            window=window,
            normalize=normalize,
            n_fft=n_fft,
            preemph=preemph,
            nfilt=features,
            lowfreq=lowfreq,
            highfreq=highfreq,
            log=log,
            log_zero_guard_type=log_zero_guard_type,
            log_zero_guard_value=log_zero_guard_value,
            dither=dither,
            pad_to=pad_to,
            frame_splicing=frame_splicing,
            exact_pad=exact_pad,
            pad_value=pad_value,
            mag_power=mag_power,
            rng=rng,
            nb_augmentation_prob=nb_augmentation_prob,
            nb_max_freq=nb_max_freq,
            mel_norm=mel_norm,
        )

    def get_features(self, input_signal, length):
        return self.featurizer(input_signal, length) # return tensor shape of (B, D, T)

    @property
    def filter_banks(self):
        return self.featurizer.filter_banks


class AudioToMFCCPreprocessor(AudioPreprocessor):
    """Preprocessor that converts wavs to MFCCs.
    Uses torchaudio.transforms.MFCC.

    Args:
        sample_rate: The sample rate of the audio.
            Defaults to 16000.
        window_size: Size of window for fft in seconds. Used to calculate the
            win_length arg for mel spectrogram.
            Defaults to 0.02
        window_stride: Stride of window for fft in seconds. Used to caculate
            the hop_length arg for mel spect.
            Defaults to 0.01
        n_window_size: Size of window for fft in samples
            Defaults to None. Use one of window_size or n_window_size.
        n_window_stride: Stride of window for fft in samples
            Defaults to None. Use one of window_stride or n_window_stride.
        window: Windowing function for fft. can be one of ['hann',
            'hamming', 'blackman', 'bartlett', 'none', 'null'].
            Defaults to 'hann'
        n_fft: Length of FT window. If None, it uses the smallest power of 2
            that is larger than n_window_size.
            Defaults to None
        lowfreq (int): Lower bound on mel basis in Hz.
            Defaults to 0
        highfreq  (int): Lower bound on mel basis in Hz.
            Defaults to None
        n_mels: Number of mel filterbanks.
            Defaults to 64
        n_mfcc: Number of coefficients to retain
            Defaults to 64
        dct_type: Type of discrete cosine transform to use
        norm: Type of norm to use
        log: Whether to use log-mel spectrograms instead of db-scaled.
            Defaults to True.
    """

    def __init__(
        self,
        sample_rate=16000,
        window_size=0.02,
        window_stride=0.01,
        n_window_size=None,
        n_window_stride=None,
        window='hann',
        n_fft=None,
        lowfreq=0.0,
        highfreq=None,
        n_mels=64,
        n_mfcc=64,
        dct_type=2,
        norm='ortho',
        log=True,
    ):
        self._sample_rate = sample_rate
        if not HAVE_TORCHAUDIO:
            logger.warning('Could not import torchaudio. Some features might not work.')

            raise ModuleNotFoundError(
                "torchaudio is not installed but is necessary for "
                "AudioToMFCCPreprocessor. We recommend you try "
                "building it from source for the PyTorch version you have."
            )
        if window_size and n_window_size:
            raise ValueError(f"{self} received both window_size and " f"n_window_size. Only one should be specified.")
        if window_stride and n_window_stride:
            raise ValueError(
                f"{self} received both window_stride and " f"n_window_stride. Only one should be specified."
            )
        # Get win_length (n_window_size) and hop_length (n_window_stride)
        if window_size:
            n_window_size = int(window_size * self._sample_rate)
        if window_stride:
            n_window_stride = int(window_stride * self._sample_rate)

        super().__init__(n_window_size, n_window_stride)

        mel_kwargs = {}

        mel_kwargs['f_min'] = lowfreq
        mel_kwargs['f_max'] = highfreq
        mel_kwargs['n_mels'] = n_mels

        mel_kwargs['n_fft'] = n_fft or 2 ** math.ceil(math.log2(n_window_size))

        mel_kwargs['win_length'] = n_window_size
        mel_kwargs['hop_length'] = n_window_stride

        # Set window_fn. None defaults to torch.ones.
        window_fn = self.torch_windows.get(window, None)
        if window_fn is None:
            raise ValueError(
                f"Window argument for AudioProcessor is invalid: {window}."
                f"For no window function, use 'ones' or None."
            )
        mel_kwargs['window_fn'] = window_fn

        # Use torchaudio's implementation of MFCCs as featurizer
        self.featurizer = torchaudio.transforms.MFCC(
            sample_rate=self._sample_rate,
            n_mfcc=n_mfcc,
            dct_type=dct_type,
            norm=norm,
            log_mels=log,
            melkwargs=mel_kwargs,
        )

    def get_features(self, input_signal, length):
        features = self.featurizer(input_signal)
        seq_len = torch.ceil(length.to(torch.float32) / self.hop_length).to(dtype=torch.long)
        return features, seq_len


class SpectrogramAugmentation(NeuralModule):
    """
    Performs time and freq cuts in one of two ways.
    SpecAugment zeroes out vertical and horizontal sections as described in
    SpecAugment (https://arxiv.org/abs/1904.08779). Arguments for use with
    SpecAugment are `freq_masks`, `time_masks`, `freq_width`, and `time_width`.
    SpecCutout zeroes out rectangulars as described in Cutout
    (https://arxiv.org/abs/1708.04552). Arguments for use with Cutout are
    `rect_masks`, `rect_freq`, and `rect_time`.

    Args:
        freq_masks (int): how many frequency segments should be cut.
            Defaults to 0.
        time_masks (int): how many time segments should be cut
            Defaults to 0.
        freq_width (int): maximum number of frequencies to be cut in one
            segment.
            Defaults to 10.
        time_width (int): maximum number of time steps to be cut in one
            segment
            Defaults to 10.
        rect_masks (int): how many rectangular masks should be cut
            Defaults to 0.
        rect_freq (int): maximum size of cut rectangles along the frequency
            dimension
            Defaults to 5.
        rect_time (int): maximum size of cut rectangles along the time
            dimension
            Defaults to 25.
        use_numba_spec_augment: use numba code for Spectrogram augmentation
        use_vectorized_spec_augment: use vectorized code for Spectrogram augmentation

    """

    def __init__(
        self,
        freq_masks=0,
        time_masks=0,
        freq_width=10,
        time_width=10,
        rect_masks=0,
        rect_time=5,
        rect_freq=20,
        rng=None,
        mask_value=0.0,
        use_vectorized_spec_augment: bool = True,
    ):
        super().__init__()

        if rect_masks > 0:
            self.spec_cutout = SpecCutout(
                rect_masks=rect_masks,
                rect_time=rect_time,
                rect_freq=rect_freq,
                rng=rng,
            )
            # self.spec_cutout.to(self._device)
        else:
            self.spec_cutout = lambda input_spec: input_spec
        if freq_masks + time_masks > 0:
            self.spec_augment = SpecAugment(
                freq_masks=freq_masks,
                time_masks=time_masks,
                freq_width=freq_width,
                time_width=time_width,
                rng=rng,
                mask_value=mask_value,
                use_vectorized_code=use_vectorized_spec_augment,
            )
        else:
            self.spec_augment = lambda input_spec, length: input_spec

    def forward(self, input_spec, length):
        augmented_spec = self.spec_cutout(input_spec=input_spec)
        augmented_spec = self.spec_augment(input_spec=augmented_spec, length=length)
        return augmented_spec # # return tensor shape of (B, D, T)