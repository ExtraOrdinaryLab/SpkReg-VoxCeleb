import os
import sys
import json
import time
import math
import logging
import warnings
from pathlib import Path
from random import randint
from functools import partial
from dataclasses import dataclass, field
from typing import List, Callable, Any, Optional

import numpy as np
from tqdm import tqdm
from rich import print
from sklearn.metrics import accuracy_score

import transformers
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, EvalPrediction
from transformers.utils import send_example_telemetry
from datasets import load_dataset, Audio, ClassLabel, Features, Value

from models.ecapa_tdnn.logging import ColoredFormatter
from models.ecapa_tdnn.configuration_ecapa_tdnn import EcapaTdnnConfig
from models.ecapa_tdnn.feature_extraction_ecapa_tdnn import EcapaTdnnFeatureExtractor
from models.ecapa_tdnn.modeling_ecapa_tdnn import EcapaTdnnForSequenceClassification

ConfigClass = EcapaTdnnConfig
FeatureExtractorClass = EcapaTdnnFeatureExtractor
ModelClass = EcapaTdnnForSequenceClassification


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    # dataset_dir: Optional[str] = field(default=None, metadata={"help": ""})
    # dataset_name: Optional[str] = field(default='confit/gtzan', metadata={"help": "Name of a dataset from the datasets package"})
    # dataset_config_name: Optional[str] = field(
    #     default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    # )
    # data_dir: Optional[str] = field(default=None, metadata={"help": ""})
    train_file: Optional[str] = field(
        default_factory=list, metadata={"help": "A file containing the training audio paths and labels."}
    )
    eval_file: Optional[str] = field(
        default=None, metadata={"help": "A file containing the validation audio paths and labels."}
    )
    train_split_name: str = field(
        default="train",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"
        },
    )
    eval_split_name: str = field(
        default="validation",
        metadata={
            "help": (
                "The name of the training data set split to use (via the datasets library). Defaults to 'validation'"
            )
        },
    )
    predict_split_name: str = field(
        default="test",
        metadata={
            "help": (
                "The name of the training data set split to use (via the datasets library). Defaults to 'test'"
            )
        },
    )
    validation_percentage: float = field(
        default=0.01,
        metadata={
            "help": "The percentage of the training data to use for validation. Defaults to 0.01."
        },
    )
    audio_column_name: str = field(
        default="audio",
        metadata={"help": "The name of the dataset column containing the audio data. Defaults to 'audio'"},
    )
    label_column_name: str = field(
        default="label", metadata={"help": "The name of the dataset column containing the labels. Defaults to 'label'"}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_length_seconds: float = field(
        default=20,
        metadata={"help": "Audio clips will be randomly cut to this length during training if the value is set."},
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from the Hub"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    feature_extractor_name: Optional[str] = field(
        default=None, metadata={"help": "Name or path of preprocessor config."}
    )
    return_attention_mask: bool = field(
        default=True, metadata={"help": "Whether to generate an attention mask in the feature extractor."}
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )
    objective: str = field(
        default='arc_face', 
    )
    angular_scale: float = field(
        default=30, 
    )
    angular_margin: float = field(
        default=0.2, 
    )


def timer_decorator(func):
    counter = 0  # To track recursion depth

    def wrapper(*args, **kwargs):
        nonlocal counter
        if counter == 0:  # Only start timing for the outermost call
            wrapper.start_time = time.time()
        counter += 1
        try:
            result = func(*args, **kwargs)  # Execute the wrapped function
        finally:
            counter -= 1
            if counter == 0:  # Only print timing for the outermost call
                end_time = time.time()
                print(f"Execution time for {func.__name__}: {end_time - wrapper.start_time:.2f} seconds")
        return result

    return wrapper


@timer_decorator
def fast_scandir(path: str, extensions: List[str], recursive: bool = False):
    # Scan files recursively faster than glob
    # From github.com/drscotthawley/aeiou/blob/main/aeiou/core.py
    subfolders, files = [], []

    try: # hope to avoid 'permission denied' by this try
        for f in os.scandir(path):
            try: # 'hope to avoid too many levels of symbolic links' error
                if f.is_dir():
                    subfolders.append(f.path)
                elif f.is_file():
                    if os.path.splitext(f.name)[1].lower() in extensions:
                        files.append(f.path)
            except Exception:
                pass
    except Exception:
        pass

    if recursive:
        for path in list(subfolders):
            sf, f = fast_scandir(path, extensions, recursive=recursive)
            subfolders.extend(sf)
            files.extend(f) # type: ignore

    return subfolders, files


def random_subsample(wav: np.ndarray, max_length: float, sample_rate: int = 16000):
    """Randomly sample chunks of `max_length` seconds from the input audio"""
    sample_length = int(round(sample_rate * max_length))
    if len(wav) <= sample_length:
        return wav
    random_offset = randint(0, len(wav) - sample_length - 1)
    return wav[random_offset : random_offset + sample_length]


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_ecapa_tdnn_voxceleb", model_args, data_args)

    # Set up logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    formatter = ColoredFormatter(datefmt="%Y-%m-%d %H:%M:%S")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to train from scratch."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    raw_datasets = load_dataset(
        'csv', 
        data_files={
            "train": data_args.train_file, 
        }, 
        delimiter="\t", 
        column_names=["audio", "label"], 
        features=Features({
            "audio": Audio(sampling_rate=16_000),
            "label": Value("string")
        }), 
        # 'data_args.dataset_name, 
        # data_dir=data_args.data_dir', 
        trust_remote_code=model_args.trust_remote_code, 
    )

    speakers = sorted(list(set(raw_datasets['train']['label'])))
    raw_datasets = raw_datasets.cast_column(
        'label', ClassLabel(num_classes=len(speakers), names=speakers)
    )

    raw_datasets = raw_datasets["train"].train_test_split(
        test_size=data_args.validation_percentage, 
        stratify_by_column='label'
    )
    raw_datasets['validation'] = raw_datasets['test']
    logger.info(f'Dataset loaded successfully: \n{raw_datasets}')
    
    num_classes = len(speakers)
    label2id, id2label = {}, {}
    for i, label in enumerate(speakers):
        label2id[label] = str(i)
        id2label[str(i)] = label

    feature_extractor = FeatureExtractorClass(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0,
        return_attention_mask=model_args.return_attention_mask,
        do_normalize=True,
    )
    model_input_name = feature_extractor.model_input_names[0]

    raw_datasets = raw_datasets.cast_column(
        data_args.audio_column_name, Audio(sampling_rate=feature_extractor.sampling_rate)
    )

    def train_transforms(batch):
        """Apply train_transforms across a batch."""
        subsampled_wavs = []
        for audio in batch[data_args.audio_column_name]:
            wav = random_subsample(
                audio["array"], max_length=data_args.max_length_seconds, sample_rate=feature_extractor.sampling_rate
            )
            subsampled_wavs.append(wav)
        inputs = feature_extractor(subsampled_wavs, sampling_rate=feature_extractor.sampling_rate)
        output_batch = {model_input_name: inputs.get(model_input_name)}
        output_batch["labels"] = list(batch['label'])
        return output_batch

    def val_transforms(batch):
        """Apply val_transforms across a batch."""
        wavs = [audio["array"] for audio in batch[data_args.audio_column_name]]
        inputs = feature_extractor(wavs, sampling_rate=feature_extractor.sampling_rate)
        output_batch = {model_input_name: inputs.get(model_input_name)}
        output_batch["labels"] = list(batch['label'])
        return output_batch
    
    raw_datasets["train"].set_transform(train_transforms, output_all_columns=False)
    
    if 'validation' in raw_datasets:
        raw_datasets["validation"].set_transform(val_transforms, output_all_columns=False)
    if 'test' in raw_datasets:
        raw_datasets["test"].set_transform(val_transforms, output_all_columns=False)
    logger.info("Datasets transformed successfully.")

    # Define our compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with
    # `predictions` and `label_ids` fields) and has to return a dictionary string to float.
    def compute_metrics(eval_pred: EvalPrediction):
        """Computes accuracy on a batch of predictions"""
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        predictions = np.argmax(logits, axis=-1)  # Convert logits to class indices
        return {
            "accuracy": accuracy_score(labels, predictions)
        }

    config = ConfigClass.from_pretrained(
        model_args.config_name, 
        num_labels=num_classes, 
        label2id=label2id,
        id2label=id2label, 
        objective=model_args.objective, 
        angular_scale=model_args.angular_scale, 
        angular_margin=model_args.angular_margin, 
    )
    model = ModelClass(config, fp16=training_args.fp16)
    logger.info(f"Model structure: \n{model}")

    if training_args.push_to_hub:
        ConfigClass.register_for_auto_class()
        FeatureExtractorClass.register_for_auto_class("AutoFeatureExtractor")
        ModelClass.register_for_auto_class("AutoModelForAudioClassification")
        config.push_to_hub(training_args.hub_model_id)
        feature_extractor.push_to_hub(training_args.hub_model_id)
        model.push_to_hub(training_args.hub_model_id)

    if training_args.do_train:
        if data_args.max_train_samples is not None:
            raw_datasets["train"] = (
                raw_datasets["train"].shuffle(seed=training_args.seed).select(range(data_args.max_train_samples))
            )
        # Set the training transforms
        raw_datasets["train"].set_transform(train_transforms, output_all_columns=False)

    if training_args.do_eval:
        if data_args.eval_split_name == 'validation':
            raw_datasets["eval"] = raw_datasets["validation"]
        if data_args.eval_split_name == 'test':
            raw_datasets["eval"] = raw_datasets["test"]
        if data_args.max_eval_samples is not None:
            raw_datasets["eval"] = (
                raw_datasets["eval"].shuffle(seed=training_args.seed).select(range(data_args.max_eval_samples))
            )
        # Set the validation transforms
        raw_datasets["eval"].set_transform(val_transforms, output_all_columns=False)

    if training_args.do_predict:
        if data_args.predict_split_name == 'validation':
            raw_datasets["predict"] = raw_datasets["validation"]
        if data_args.predict_split_name == 'test':
            raw_datasets["predict"] = raw_datasets["test"]
        # Set the test transforms
        raw_datasets["predict"].set_transform(val_transforms, output_all_columns=False)

    # Initialize our trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=raw_datasets["train"] if training_args.do_train else None,
        eval_dataset=raw_datasets["eval"] if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        processing_class=feature_extractor,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

        log_history = trainer.state.log_history
        log_history_path = os.path.join(training_args.output_dir, 'log_history.json')
        with open(log_history_path, "w") as f:
            json.dump(log_history, f, indent=4)

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        metrics = trainer.evaluate(raw_datasets["predict"], metric_key_prefix="predict")
        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

    # Write model card and (optionally) push to hub
    kwargs = {
        # "finetuned_from": 'confit/xvector-voxceleb1',
        "tasks": "audio-classification",
        # "dataset": data_args.dataset_name,
        "tags": ["audio-classification"],
    }
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == '__main__':
    main()