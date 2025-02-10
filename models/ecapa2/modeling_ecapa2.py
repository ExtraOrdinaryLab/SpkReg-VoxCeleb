from dataclasses import dataclass
from typing import Optional, Union, Tuple

import torch
import torch.nn as nn

from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from .configuration_ecapa2 import Ecapa2Config
from .audio_processing import AudioToMelSpectrogramPreprocessor
from .audio_processing import SpectrogramAugmentation
from .conv_asr import Ecapa2Encoder, SpeakerDecoder
from .angular_loss import AdditiveMarginSoftmaxLoss, AdditiveAngularMarginSoftmaxLoss


@dataclass
class Ecapa2BaseModelOutput(ModelOutput):

    encoder_outputs: torch.FloatTensor = None
    extract_features: torch.FloatTensor = None
    output_lengths: torch.FloatTensor = None


@dataclass
class Ecapa2SequenceClassifierOutput(ModelOutput):

    loss: torch.FloatTensor = None
    logits: torch.FloatTensor = None
    embeddings: torch.FloatTensor = None


class Ecapa2PreTrainedModel(PreTrainedModel):

    config_class = Ecapa2Config
    base_model_prefix = "ecapa2"
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


class Ecapa2Model(Ecapa2PreTrainedModel):

    def __init__(self, config: Ecapa2Config):
        super().__init__(config)
        self.config = config

        self.preprocessor = AudioToMelSpectrogramPreprocessor(**config.mel_spectrogram_config)
        self.spec_augment = SpectrogramAugmentation(**config.spectrogram_augmentation_config)
        self.encoder = Ecapa2Encoder(**config.encoder_config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None, 
    ) -> Union[Tuple, Ecapa2BaseModelOutput]:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_values).to(input_values)
        lengths = attention_mask.sum(dim=1).long()
        extract_features, output_lengths = self.preprocessor(input_values, lengths)
        if self.training:
            extract_features = self.spec_augment(extract_features, output_lengths)
        encoder_outputs, output_lengths = self.encoder(extract_features, output_lengths)

        return Ecapa2BaseModelOutput(
            encoder_outputs=encoder_outputs, 
            extract_features=extract_features, 
            output_lengths=output_lengths, 
        )


class Ecapa2ForSequenceClassification(Ecapa2PreTrainedModel):

    def __init__(self, config: Ecapa2Config):
        super().__init__(config)

        self.ecapa2 = Ecapa2Model(config)
        self.classifier = SpeakerDecoder(**config.decoder_config)

        if config.objective == 'additive_angular_margin':
            self.loss_fct = AdditiveAngularMarginSoftmaxLoss(**config.objective_config)
        elif config.objective == 'additive_margin':
            self.loss_fct = AdditiveMarginSoftmaxLoss(**config.objective_config)
        elif config.objective == 'cross_entropy':
            self.loss_fct = nn.CrossEntropyLoss(**config.objective_config)

        self.init_weights()

    def freeze_base_model(self):
        for param in self.ecapa2.parameters():
            param.requires_grad = False

    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, Ecapa2SequenceClassifierOutput]:
        ecapa2_outputs = self.ecapa2(
            input_values, 
            attention_mask, 
        )
        logits, output_embeddings = self.classifier(
            ecapa2_outputs.encoder_outputs, 
            ecapa2_outputs.output_lengths
        )
        logits = logits.view(-1, self.config.num_labels)

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels.view(-1))

        return Ecapa2SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            embeddings=output_embeddings, 
        )