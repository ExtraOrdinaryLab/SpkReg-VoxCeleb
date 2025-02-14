from models.ecapa_tdnn.configuration_ecapa_tdnn import EcapaTdnnConfig


# ECAPA-TDNN channel=512 configuration
config = EcapaTdnnConfig(
    features=80, 
    sample_rate=16_000, 
    window_size=0.025, 
    window_stride=0.01, 
    n_fft=512, 
    freq_masks=3, 
    freq_width=4, 
    time_masks=5, 
    time_width=0.03, 
    filters=[512, 512, 512, 512, 1536], 
    kernel_sizes=[5, 3, 3, 3, 1], 
    dilations=[1, 2, 3, 4, 1], 
    res2net_scale=8, 
    pool_mode='attention', # xvector, tap or attention
    emb_sizes=192, 
    objective='additive_angular_margin', 
    implementation='speechbrain', 
)
config.save_pretrained('configs/c512')


# ECAPA-TDNN channel=1024 configuration
config = EcapaTdnnConfig(
    features=80, 
    sample_rate=16_000, 
    window_size=0.025, 
    window_stride=0.01, 
    n_fft=512, 
    freq_masks=3, 
    freq_width=4, 
    time_masks=5, 
    time_width=0.03, 
    filters=[1024, 1024, 1024, 1024, 3072], 
    kernel_sizes=[5, 3, 3, 3, 1], 
    dilations=[1, 2, 3, 4, 1], 
    res2net_scale=8, 
    pool_mode='attention', # xvector, tap or attention
    emb_sizes=192, 
    objective='additive_angular_margin', 
    implementation='speechbrain', 
)
config.save_pretrained('configs/c1024')