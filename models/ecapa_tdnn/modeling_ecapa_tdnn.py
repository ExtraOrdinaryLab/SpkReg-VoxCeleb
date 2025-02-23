from dataclasses import dataclass
from typing import Optional, Union, Tuple

import torch
import torch.nn as nn

from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from .configuration_ecapa_tdnn import EcapaTdnnConfig
from .audio_processing import AudioToMelSpectrogramPreprocessor
from .audio_processing import SpectrogramAugmentation
from .conv_asr import SpeakerDecoder
from .speechbrain.ecapa_tdnn_encoder import EcapaTdnnEncoder as SpeechBrainEcapaTdnnEncoder
from .wespeaker.ecapa_tdnn_encoder import EcapaTdnnEncoder as WeSpeakerEcapaTdnnEncoder
from .nemo.ecapa_tdnn_encoder import EcapaTdnnEncoder as NeMoEcapaTdnnEncoder
from .angular_loss import (
    NeMoArcFaceLoss, 
    SpeechBrainArcFaceLoss, 
    CosFaceLoss, 
    ArcFaceLoss, 
    SphereFaceLoss, 
    AdaCosLoss, 
    ChebyshevArcFaceLoss
)


@dataclass
class EcapaTdnnBaseModelOutput(ModelOutput):

    encoder_outputs: torch.FloatTensor = None
    extract_features: torch.FloatTensor = None
    output_lengths: torch.FloatTensor = None


@dataclass
class EcapaTdnnSequenceClassifierOutput(ModelOutput):

    loss: torch.FloatTensor = None
    logits: torch.FloatTensor = None
    embeddings: torch.FloatTensor = None


class EcapaTdnnPreTrainedModel(PreTrainedModel):

    config_class = EcapaTdnnConfig
    base_model_prefix = "ecapa_tdnn"
    main_input_name = "input_values"

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Conv2d):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    @property
    def num_weights(self):
        """
        Utility property that returns the total number of parameters of NeuralModule.
        """
        return self._num_weights()

    @torch.jit.ignore
    def _num_weights(self):
        num: int = 0
        for p in self.parameters():
            if p.requires_grad:
                num += p.numel()
        return num


class EcapaTdnnModel(EcapaTdnnPreTrainedModel):

    def __init__(self, config: EcapaTdnnConfig):
        super().__init__(config)
        self.config = config

        self.preprocessor = AudioToMelSpectrogramPreprocessor(
            features=config.features, 
            sample_rate=config.sample_rate, 
            window_size=config.window_size, 
            window_stride=config.window_stride, 
            n_fft=config.n_fft, 
        )
        self.spec_augment = SpectrogramAugmentation(
            freq_masks=config.freq_masks, 
            time_masks=config.time_masks, 
            freq_width=config.freq_width, 
            time_width=config.time_width, 
            rect_masks=config.rect_masks, 
            rect_time=config.rect_time,
            rect_freq=config.rect_freq, 
            mask_value=config.mask_value,
        )

        if config.implementation == 'speechbrain':
            EcapaTdnnEncoder = SpeechBrainEcapaTdnnEncoder
        elif config.implementation == 'wespeaker':
            EcapaTdnnEncoder = WeSpeakerEcapaTdnnEncoder
        elif config.implementation == 'nemo':
            EcapaTdnnEncoder = NeMoEcapaTdnnEncoder

        self.encoder = EcapaTdnnEncoder(
            features=config.features, 
            filters=config.filters, 
            kernel_sizes=config.kernel_sizes, 
            dilations=config.dilations, 
            res2net_scale=config.res2net_scale,
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None, 
    ) -> Union[Tuple, EcapaTdnnBaseModelOutput]:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_values).to(input_values)
        lengths = attention_mask.sum(dim=1).long()
        extract_features, output_lengths = self.preprocessor(input_values, lengths)
        if self.training:
            extract_features = self.spec_augment(extract_features, output_lengths)
        encoder_outputs, output_lengths = self.encoder(extract_features, output_lengths)

        return EcapaTdnnBaseModelOutput(
            encoder_outputs=encoder_outputs, 
            extract_features=extract_features, 
            output_lengths=output_lengths, 
        )


class EcapaTdnnForSequenceClassification(EcapaTdnnPreTrainedModel):

    def __init__(self, config: EcapaTdnnConfig, fp16: bool = False):
        super().__init__(config)
        self.config = config

        self.ecapa_tdnn = EcapaTdnnModel(config)
        self.classifier = SpeakerDecoder(
            feat_in=config.filters[-1], 
            num_classes=config.num_labels, 
            emb_sizes=config.emb_sizes, 
            pool_mode=config.pool_mode, 
            angular=config.angular, 
            attention_channels=config.attention_channels, 
            init_mode=config.init_mode, 
        )

        if config.objective in ['additive_angular_margin', 'arc_face']:
            self.loss_fct = ArcFaceLoss(
                scale=config.angular_scale, 
                margin=config.angular_margin, 
            )
        elif config.objective in ['nemo_arc_face']:
            self.loss_fct = NeMoArcFaceLoss(
                scale=config.angular_scale, 
                margin=config.angular_margin, 
            )
        elif config.objective in ['speechbrain_arc_face']:
            self.loss_fct = SpeechBrainArcFaceLoss(
                scale=config.angular_scale, 
                margin=config.angular_margin, 
                dtype=(torch.float16 if fp16 else torch.float32)
            )
        elif config.objective in ['additive_margin', 'cos_face']:
            self.loss_fct = CosFaceLoss(
                scale=config.angular_scale, 
                margin=config.angular_margin, 
            )
        elif config.objective in ['multiplicative_angular_margin', 'sphere_face']:
            self.loss_fct = SphereFaceLoss(
                scale=config.angular_scale, 
                margin=config.angular_margin, 
            )
        elif config.objective in ['adaptive_margin', 'ada_cos']:
            self.loss_fct = AdaCosLoss(
                initial_scale=config.angular_scale, 
            )
        elif config.objective in ['chebyshev_arc_face']:
            self.loss_fct = ChebyshevArcFaceLoss(
                scale=config.angular_scale, 
                margin=config.angular_margin,
                chebyshev_degree=10, 
            )
        elif config.objective == 'cross_entropy':
            self.loss_fct = nn.CrossEntropyLoss(
                label_smoothing=config.label_smoothing
            )

        self.init_weights()

    def freeze_base_model(self):
        for param in self.ecapa_tdnn.parameters():
            param.requires_grad = False

    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, EcapaTdnnSequenceClassifierOutput]:
        ecapa_tdnn_outputs = self.ecapa_tdnn(
            input_values, 
            attention_mask, 
        )
        logits, output_embeddings = self.classifier(
            ecapa_tdnn_outputs.encoder_outputs, 
            ecapa_tdnn_outputs.output_lengths
        )
        logits = logits.view(-1, self.config.num_labels)

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels.view(-1))

        return EcapaTdnnSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            embeddings=output_embeddings, 
        )