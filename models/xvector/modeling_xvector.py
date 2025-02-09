from dataclasses import dataclass
from typing import Optional, Union, Tuple

from rich import print

import torch
import torch.nn as nn

from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from .configuration_xvector import XVectorConfig
from .audio_processing import AudioToMelSpectrogramPreprocessor
from .audio_processing import SpectrogramAugmentation
from .conv_asr import XVectorEncoder, SpeakerDecoder
from .angular_loss import AdditiveMarginSoftmaxLoss, AdditiveAngularMarginSoftmaxLoss


@dataclass
class XVectorBaseModelOutput(ModelOutput):

    encoder_outputs: torch.FloatTensor = None
    extract_features: torch.FloatTensor = None
    output_lengths: torch.FloatTensor = None


@dataclass
class XVectorSequenceClassifierOutput(ModelOutput):

    loss: torch.FloatTensor = None
    logits: torch.FloatTensor = None
    embeddings: torch.FloatTensor = None


class XVectorPreTrainedModel(PreTrainedModel):

    config_class = XVectorConfig
    base_model_prefix = "xvector"
    main_input_name = "input_values"

    def _init_weights(self, module):
        """Initialize the weights"""
        config: XVectorConfig = self.config
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Conv2d):
            module.weight.data.normal_(mean=0.0, std=config.initializer_range)
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


class XVectorModel(XVectorPreTrainedModel):

    def __init__(self, config: XVectorConfig):
        super().__init__(config)
        self.config = config

        self.preprocessor = AudioToMelSpectrogramPreprocessor(**config.mel_spectrogram_config)
        self.spec_augment = SpectrogramAugmentation(**config.spectrogram_augmentation_config)
        self.encoder = XVectorEncoder(**config.encoder_config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None, 
    ) -> Union[Tuple, XVectorBaseModelOutput]:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_values).to(input_values)
        lengths = attention_mask.sum(dim=1).long()
        extract_features, output_lengths = self.preprocessor(input_values, lengths)
        if self.training:
            extract_features = self.spec_augment(extract_features, output_lengths)
        encoder_outputs, output_lengths = self.encoder(extract_features, output_lengths)

        return XVectorBaseModelOutput(
            encoder_outputs=encoder_outputs, 
            extract_features=extract_features, 
            output_lengths=output_lengths, 
        )


class XVectorForSequenceClassification(XVectorPreTrainedModel):

    def __init__(self, config: XVectorConfig):
        super().__init__(config)

        self.xvector = XVectorModel(config)
        self.classifier = SpeakerDecoder(**config.decoder_config)

        if config.objective == 'additive_angular_margin':
            self.loss_fct = AdditiveAngularMarginSoftmaxLoss(**config.objective_config)
        elif config.objective == 'additive_margin':
            self.loss_fct = AdditiveMarginSoftmaxLoss(**config.objective_config)
        elif config.objective == 'cross_entropy':
            self.loss_fct = nn.CrossEntropyLoss(**config.objective_config)

        self.init_weights()

    def freeze_base_model(self):
        for param in self.xvector.parameters():
            param.requires_grad = False

    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, XVectorSequenceClassifierOutput]:
        xvector_outputs = self.xvector(
            input_values, 
            attention_mask, 
        )
        logits, output_embeddings = self.classifier(
            xvector_outputs.encoder_outputs, 
            xvector_outputs.output_lengths
        )
        logits = logits.view(-1, self.config.num_labels)

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels.view(-1))

        return XVectorSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            embeddings=output_embeddings, 
        )