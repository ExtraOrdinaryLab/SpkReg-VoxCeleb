import os
import sys
import json
import time
import math
import logging
import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm
from rich import print
from sklearn import metrics

import torch
from datasets import load_dataset, Audio, ClassLabel, Features, Value

from models.ecapa_tdnn.logging import ColoredFormatter
from models.ecapa_tdnn.configuration_ecapa_tdnn import EcapaTdnnConfig
from models.ecapa_tdnn.feature_extraction_ecapa_tdnn import EcapaTdnnFeatureExtractor
from models.ecapa_tdnn.modeling_ecapa_tdnn import EcapaTdnnForSequenceClassification

ConfigClass = EcapaTdnnConfig
FeatureExtractorClass = EcapaTdnnFeatureExtractor
ModelClass = EcapaTdnnForSequenceClassification


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--eval_file", default=None, nargs='+', help="")
    parser.add_argument("--model_name_or_path", type=str, default=None, help="")
    parser.add_argument("--trial_file", nargs='+', help="")
    parser.add_argument("--device", type=str, default='cuda:0', help="")
    parser.add_argument("--trust_remote_code", action="store_true", help="")
    return parser.parse_args()


def eer_score(positive_scores, negative_scores, return_threshold=False):
    from scipy.optimize import brentq
    from scipy.interpolate import interp1d

    scores = positive_scores + negative_scores
    labels = [1] * len(positive_scores) + [0] * len(negative_scores)
    fpr, tpr, thresholds_ = metrics.roc_curve(labels, scores, pos_label=1)

    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresholds = interp1d(fpr, thresholds_)(eer)
    
    if return_threshold:
        return float(eer), float(thresholds)
    return float(eer)


def mindcf_score(positive_scores, negative_scores, p_target=0.01, c_miss=1.0, c_fa=1.0, return_threshold=False):
    scores = positive_scores + negative_scores
    labels = [1] * len(positive_scores) + [0] * len(negative_scores)
    fprs, tprs, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    fnrs = 1 - tprs
    min_c_det = float("inf")
    min_c_det_threshold = thresholds[0]
    for i in range(0, len(fnrs)):
        # See Equation (2).  it is a weighted sum of false negative
        # and false positive errors.
        c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]
    # See Equations (3) and (4).  Now we normalize the cost.
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def

    if return_threshold:
        return float(min_dcf), float(min_c_det_threshold)
    return min_dcf


def main(args):
    # Set up logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    formatter = ColoredFormatter(datefmt="%Y-%m-%d %H:%M:%S")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    feature_extractor = FeatureExtractorClass.from_pretrained(
        args.model_name_or_path, 
    )
    model_input_name = feature_extractor.model_input_names[0]

    raw_datasets = load_dataset(
        'csv', 
        data_files={
            "test": args.eval_file, 
        }, 
        delimiter="\t", 
        column_names=["audio", "label"], 
        features=Features({
            "audio": Audio(sampling_rate=feature_extractor.sampling_rate),
            "label": Value("string")
        }), 
        trust_remote_code=args.trust_remote_code, 
    )
    test_dataset = raw_datasets['test']
    logger.info(f'Dataset loaded successfully: \n{test_dataset}')

    config = ConfigClass.from_pretrained(
        args.model_name_or_path, 
    )
    model = ModelClass.from_pretrained(
        args.model_name_or_path, config=config, fp16=False
    )
    model.to(args.device)
    logger.info(f"Model structure: \n{model}")

    test_dataset = test_dataset.cast_column(
        'audio', Audio(sampling_rate=feature_extractor.sampling_rate)
    )

    filename2embedding = {}
    for example in tqdm(test_dataset, desc='Compute Embeddings'):
        filepath = example['audio']['path']
        dir_path = Path(filepath).parents[2] # get directory two levels up
        filename = os.path.relpath(filepath, dir_path)
        
        inputs = feature_extractor(
            example['audio']['array'], 
            sampling_rate=feature_extractor.sampling_rate, 
            return_tensors='pt'
        )
        inputs = {k: v.to(args.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.embeddings

        # filename: id10270/x6uYqmx31kE/00001.wav
        # embedding: tensor shape of (1, D)
        filename2embedding[filename] = embeddings.to('cpu')
    
    similarity_fn = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

    for trial_file in args.trial_file:
        assert Path(trial_file).name in ['veri_test2.txt', 'list_test_hard2.txt', 'list_test_all2.txt']
        if Path(trial_file).name == 'veri_test2.txt':
            trial_name = 'voxceleb1-o'
        elif Path(trial_file).name == 'list_test_hard2.txt':
            trial_name = 'voxceleb1-h'
        elif Path(trial_file).name == 'list_test_all2.txt':
            trial_name = 'voxceleb1-e'

        # Load the verification trial file into a list
        with open(trial_file, "r") as f:
            trial_list = [line.strip().split() for line in f]
        trial_list = [(int(parts[0]), parts[1], parts[2]) for parts in trial_list]
    
        positive_scores, negative_scores = [], []
        for parts in tqdm(trial_list, desc='Trial'):
            label, enrol_filename, test_filename = parts
            enrol_embedding = filename2embedding[enrol_filename]
            test_embedding = filename2embedding[test_filename]

            score = similarity_fn(enrol_embedding, test_embedding)[0]

            if label == 1:
                positive_scores.append(score)
            else:
                negative_scores.append(score)

        eer = eer_score(positive_scores, negative_scores, return_threshold=False)
        mindcf = mindcf_score(positive_scores, negative_scores, p_target=0.01, c_miss=1.0, c_fa=1.0, return_threshold=False)
        
        logger.info(f"{trial_name}: EER = {eer:.6f}, minDCF = {mindcf:.6f}")


if __name__ == "__main__":
    args = parse_args()
    main(args)