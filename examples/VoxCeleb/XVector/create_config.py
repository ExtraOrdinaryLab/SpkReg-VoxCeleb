from models.xvector.configuration_xvector import XVectorConfig


# XVector (TDNN) configuration
config = XVectorConfig(
    features=80, 
    sample_rate=16_000, 
    window_size=0.025, 
    window_stride=0.01, 
    n_fft=512, 
    freq_masks=3, 
    freq_width=4, 
    time_masks=5, 
    time_width=0.03, 
    filters=[512, 512, 512, 512, 1500], 
    kernel_sizes=[5, 3, 3, 1, 1], 
    dilations=[1, 2, 3, 1, 1], 
    pool_mode='xvector', # xvector, tap or attention
    emb_sizes=[512, 512], 
    objective='cross_entropy', 
)
config.save_pretrained('configs/xvector')
