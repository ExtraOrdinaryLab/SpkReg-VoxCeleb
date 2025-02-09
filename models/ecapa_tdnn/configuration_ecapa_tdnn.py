from typing import Any, Union

from transformers.configuration_utils import PretrainedConfig


class ECAPATDNNConfig(PretrainedConfig):

    def __init__(
        self, 
        sample_rate: int = 16000,
        window_size: float = 0.02,
        window_stride: float = 0.01,
        n_window_size: Any = None,
        n_window_stride: Any = None,
        window: str = "hann",
        normalize: str = "per_feature",
        n_fft: Any = None,
        preemph: float = 0.97,
        features: int = 64,
        lowfreq: int = 0,
        highfreq: Any = None,
        log: bool = True,
        log_zero_guard_type: str = "add",
        log_zero_guard_value: Any = 2 ** -24,
        dither: float = 0.00001,
        pad_to: int = 16,
        frame_splicing: int = 1,
        exact_pad: bool = False,
        pad_value: int = 0,
        mag_power: float = 2,
        rng: Any = None,
        nb_augmentation_prob: float = 0,
        nb_max_freq: int = 4000,
        use_torchaudio: bool = True,
        mel_norm: str = "slaney", 
        freq_masks: int = 0,
        time_masks: int = 0,
        freq_width: int = 10,
        time_width: int = 10,
        rect_masks: int = 0,
        rect_time: int = 5,
        rect_freq: int = 20,
        mask_value: float = 0,
        use_vectorized_spec_augment: bool = True, 
        filters: list = [512, 512, 512, 512, 1500],
        kernel_sizes: list = [5, 3, 3, 1, 1],
        dilations: list = [1, 2, 3, 1, 1],
        scale: int = 8,
        init_mode: str = 'xavier_uniform', 
        emb_sizes: Union[int, list] = 256,
        pool_mode: str = 'xvector',
        angular: bool = False,
        attention_channels: int = 128,
        objective: str = 'angular', # angular, cross_entropy
        angular_scale = 30, 
        angular_margin: float = 0.2, 
        label_smoothing: float = 0.0, 
        initializer_range=0.02,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        **kwargs,
    ):
        super().__init__(**kwargs, pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id)

        self.initializer_range = initializer_range

        # Mel-spectrogram configuration
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.window_stride = window_stride
        self.n_window_size = n_window_size
        self.n_window_stride = n_window_stride
        self.window = window
        self.normalize = normalize
        self.n_fft = n_fft
        self.preemph = preemph
        self.features = features
        self.lowfreq = lowfreq
        self.highfreq = highfreq
        self.log = log
        self.log_zero_guard_type = log_zero_guard_type
        self.log_zero_guard_value = log_zero_guard_value
        self.dither = dither
        self.pad_to = pad_to
        self.frame_splicing = frame_splicing
        self.exact_pad = exact_pad
        self.pad_value = pad_value
        self.mag_power = mag_power
        self.rng = rng
        self.nb_augmentation_prob = nb_augmentation_prob
        self.nb_max_freq = nb_max_freq
        self.use_torchaudio = use_torchaudio
        self.mel_norm = mel_norm
        self.mel_spectrogram_config = {
            "sample_rate": sample_rate,
            "window_size": window_size, 
            "window_stride": window_stride,
            "n_window_size": n_window_size,
            "n_window_stride": n_window_stride,
            "window": window,
            "normalize": normalize,
            "n_fft": n_fft,
            "preemph": preemph,
            "features": features,
            "lowfreq": lowfreq,
            "highfreq": highfreq,
            "log": log,
            "log_zero_guard_type": log_zero_guard_type,
            "log_zero_guard_value": log_zero_guard_value,
            "dither": dither,
            "pad_to": pad_to,
            "frame_splicing": frame_splicing,
            "exact_pad": exact_pad,
            "pad_value": pad_value,
            "mag_power": mag_power,
            "rng": rng,
            "nb_augmentation_prob": nb_augmentation_prob,
            "nb_max_freq": nb_max_freq,
            "use_torchaudio": use_torchaudio,
            "mel_norm": mel_norm,
        }

        # Spectrogram Augmentation configuration
        self.freq_masks = freq_masks
        self.time_masks = time_masks
        self.freq_width = freq_width
        self.time_width = time_width
        self.rect_masks = rect_masks
        self.rect_time = rect_time
        self.rect_freq = rect_freq
        self.mask_value = mask_value
        self.use_vectorized_spec_augment = use_vectorized_spec_augment
        self.spectrogram_augmentation_config = {
            "freq_masks": freq_masks, 
            "time_masks": time_masks,
            "freq_width": freq_width,
            "time_width": time_width,
            "rect_masks": rect_masks,
            "rect_time": rect_time,
            "rect_freq": rect_freq,
            "mask_value": mask_value,
            "use_vectorized_spec_augment": use_vectorized_spec_augment,
        }

        # Encoder configuration
        self.feat_in = features
        self.filters = filters
        self.kernel_sizes = kernel_sizes
        self.dilations = dilations
        self.scale = scale
        self.init_mode = init_mode
        self.encoder_config = {
            "feat_in": features,
            "filters": filters,
            "kernel_sizes": kernel_sizes,
            "dilations": dilations,
            "scale": scale,
            "init_mode": init_mode,
        }

        # Decoder configuration
        self.emb_sizes = emb_sizes
        self.pool_mode = pool_mode
        self.angular = True if objective == 'angular' else False
        self.attention_channels = attention_channels
        self.decoder_config = {
            "feat_in": filters[-1],
            "num_classes": self.num_labels,
            "emb_sizes": emb_sizes,
            "pool_mode": pool_mode,
            "angular": angular,
            "attention_channels": attention_channels,
            "init_mode": init_mode, 
        }

        # Loss function configuration
        self.objective = objective
        if objective == 'angular':
            self.objective_config = {
                "scale": angular_scale, 
                "margin": angular_margin, 
            }
        elif objective == 'cross_entropy':
            self.objective_config = {
                "label_smoothing": label_smoothing, 
            }