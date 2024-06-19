# export PYTHONPATH="/Users/kar/Repos/X-IAA/"

.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = X-IAA
PYTHON_INTERPRETER = python3.9

PYTHONPATH  := /Users/kar/Repos/X-IAA:$(PATH)
SHELL := /bin/bash

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	clear;

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	clear;


## Utils and shared scripts

dataframe:
	$(PYTHON_INTERPRETER) src/utils/dataframe_preparation.py

dataset-split:
	$(PYTHON_INTERPRETER) src/utils/dataset_preparation.py
poc:
	$(PYTHON_INTERPRETER) src/assisted-culling-poc/sort-custom-images.py

save-distortions:
	$(PYTHON_INTERPRETER) src/utils/distortion_generator.py

extract-eva-votes:
	$(PYTHON_INTERPRETER) src/utils/extract_eva_votes_per_feature.py

## NIMA

nima-infer:
	$(PYTHON_INTERPRETER) src/nima/giiaa/inference_giiaa_histogram.py

nima-sort-images:
	$(PYTHON_INTERPRETER) src/nima/giiaa/sort_images.py

nima-save-histograms:
	clear;
	$(PYTHON_INTERPRETER) src/nima/giiaa/save_histograms.py

nima-evaluate-high-low:
	clear;
	$(PYTHON_INTERPRETER) src/nima/giiaa/evaluation_giiaa_histogram.py

nima-evaluate-distortions:
	$(PYTHON_INTERPRETER) src/nima/gciaa/evaluation_gciaa_distortions.py

nima-personalize-distortions:
	$(PYTHON_INTERPRETER) src/nima/gciaa/training_gciaa_distortions.py

nima-personalize:
	$(PYTHON_INTERPRETER) src/nima/pciaa/training_pciaa_v2.py

nima-get-histograms:
	$(PYTHON_INTERPRETER) src/nima/giiaa/evaluation_get_histograms.py

nima-train-giiaa:
	$(PYTHON_INTERPRETER) src/nima/giiaa/training_giiaa_histogram.py

nima-train-gciaa:
	$(PYTHON_INTERPRETER) src/nima/gciaa/training_gciaa_distortions.py

nima-train-contextual:
	$(PYTHON_INTERPRETER) src/nima/gciaa/training_gciaa_contextual.py

nima-infer-bw:
	$(PYTHON_INTERPRETER) src/nima/gciaa/inference_gciaa_bw.py

nima-infer-bl:
	$(PYTHON_INTERPRETER) src/nima/gciaa/inference_gciaa_bl.py

nima-infer-bo:
	$(PYTHON_INTERPRETER) src/nima/gciaa/inference_gciaa_bo.py

nima-infer-smiles:
	$(PYTHON_INTERPRETER) src/nima/gciaa/inference_gciaa_smiles.py

nima-evaluate:
	$(PYTHON_INTERPRETER) src/nima/gciaa/evaluation_gciaa_v2.py

## Brightness predictor

bp-train:
	$(PYTHON_INTERPRETER) src/brightness_predictor/training_bp.py

bp-infer:
	$(PYTHON_INTERPRETER) src/brightness_predictor/inference_bp.py

bp-explain:
	$(PYTHON_INTERPRETER) src/explanations/lime_bp.py

## MusiQ

musiq-sort-images:
	$(PYTHON_INTERPRETER) src/musiq/sort_images.py

musiq-evaluate-high-low:
	$(PYTHON_INTERPRETER) src/musiq/evaluation_high_low_quality.py

musiq-evaluate-distortions:
	$(PYTHON_INTERPRETER) src/musiq/evaluation_distortions.py

musiq-get-histograms:
	$(PYTHON_INTERPRETER) src/musiq/evaluation_get_histograms.py

musiq-train-giiaa:
	$(PYTHON_INTERPRETER) src/musiq/train_giiaa.py

## Culling PoC

culling-poc:
	$(PYTHON_INTERPRETER) src/assisted-culling-poc/sort_groups.py

## Global Explanations

tsne-nima:
	$(PYTHON_INTERPRETER) src/explanations/tsne_nima.py

tsne-musiq:
	$(PYTHON_INTERPRETER) src/explanations/tsne_musiq.py

tsne-instances:
	$(PYTHON_INTERPRETER) src/explanations/tsne_with_instances.py

tsne-instances-smiles:
	$(PYTHON_INTERPRETER) src/explanations/tsne_with_instances_smiles.py

## Local Explanations

lime-bp:
	$(PYTHON_INTERPRETER) src/explanations/lime_bp.py

lime-nima:
	$(PYTHON_INTERPRETER) src/explanations/lime_nima.py

lime-nima-draw:
	$(PYTHON_INTERPRETER) src/explanations/lime_nima_draw.py

lime-musiq:
	$(PYTHON_INTERPRETER) src/explanations/lime_musiq.py

lime-personalized:
	$(PYTHON_INTERPRETER) src/explanations/lime_personalized.py

lime-generative:
	$(PYTHON_INTERPRETER) src/explanations/lime_generative.py

lime-show-perturbations:
	$(PYTHON_INTERPRETER) src/explanations/lime_show_perturbations.py

lime-difference:
	$(PYTHON_INTERPRETER) src/explanations/lime_difference.py

lime-difference-handcraft:
	$(PYTHON_INTERPRETER) src/explanations/lime_difference_handcraft.py

get-ava-histograms:
	$(PYTHON_INTERPRETER) src/explanations/ava_histogram_comparison.py