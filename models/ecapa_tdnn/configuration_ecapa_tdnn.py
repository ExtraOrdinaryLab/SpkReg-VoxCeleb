from typing import Any, Union

from transformers.configuration_utils import PretrainedConfig


class EcapaTdnnConfig(PretrainedConfig):

    angular_losses = [
        'norm_face', 
        'additive_angular_margin', 'arc_face', 'nemo_arc_face', 'speechbrain_arc_face', 
        'additive_margin', 'cos_face', 
        'multiplicative_angular_margin', 'sphere_face', 
        'adaptive_margin', 'ada_cos', 
        'quadratic_additive_angular_margin', 'qam_face', 
        'chebyshev_arc_face', 'remez_arc_face', 'legendre_arc_face', 'jacobi_arc_face'
    ]

    def __init__(
        self, 
        features: int = 64,
        sample_rate: int = 16000,
        window_size: float = 0.025,
        window_stride: float = 0.01,
        n_fft: Any = None,
        freq_masks: int = 0,
        time_masks: int = 0,
        freq_width: int = 10,
        time_width: int = 10,
        rect_masks: int = 0,
        rect_time: int = 5,
        rect_freq: int = 20,
        mask_value: float = 0, 
        filters: list = [512, 512, 512, 512, 1500],
        kernel_sizes: list = [5, 3, 3, 1, 1],
        dilations: list = [1, 2, 3, 1, 1],
        res2net_scale: int = 8, 
        init_mode: str = 'xavier_uniform', 
        emb_sizes: Union[int, list] = 192,
        pool_mode: str = 'xvector', 
        attention_channels: int = 128,
        objective: str = 'additive_angular_margin', # additive_margin, additive_angular_margin, cross_entropy
        angular_scale = 30, 
        angular_margin: float = 0.2, 
        label_smoothing: float = 0.0, 
        initializer_range=0.02,
        implementation="speechbrain", 
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        **kwargs,
    ):
        super().__init__(**kwargs, pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id)

        self.implementation = implementation
        self.initializer_range = initializer_range

        # Mel-spectrogram configuration
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.window_stride = window_stride
        self.n_fft = n_fft
        self.features = features

        # Spectrogram Augmentation configuration
        self.freq_masks = freq_masks
        self.time_masks = time_masks
        self.freq_width = freq_width
        self.time_width = time_width
        self.rect_masks = rect_masks
        self.rect_time = rect_time
        self.rect_freq = rect_freq
        self.mask_value = mask_value

        # Encoder configuration
        self.feat_in = features
        self.filters = filters
        self.kernel_sizes = kernel_sizes
        self.dilations = dilations
        self.res2net_scale = res2net_scale
        self.init_mode = init_mode

        # Decoder configuration
        self.emb_sizes = emb_sizes
        self.pool_mode = pool_mode
        self.angular = True if objective in self.angular_losses else False
        self.attention_channels = attention_channels

        # Loss function configuration
        self.objective = objective
        self.angular_scale = angular_scale
        self.angular_margin = angular_margin
        self.label_smoothing = label_smoothing
        assert objective in self.angular_losses