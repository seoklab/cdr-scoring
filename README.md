# cdr-scoring

A machine-learning framework for evaluating and ranking predicted antibody CDR structures.

This project develops SE(3)-equivariant graph neural network models to assess CDR structural quality and prioritize near-native conformations from sets of predicted antibody structures. The framework combines local CDR geometry and antibody-antigen interface information, and supports pairwise ranking objectives for model selection.

## Main Features

- SE(3)-equivariant graph neural network for antibody structure scoring
- CDR-level structural quality assessment
- Antibody-antigen interface quality modeling
- Pairwise ranking of structural decoys
- Training and evaluation across multiple structure-generation methods
- Tools for preprocessing, training, inference, and benchmarking

## Project Structure

- `src/` - model and training code
- `scripts/` - preprocessing, training, and evaluation scripts
- `configs/` - experiment configurations
- `tests/` - test and validation code

## Status

This repository contains research code developed for ongoing work on antibody CDR structure quality assessment and ranking.

## Author

Sujin Park  
Seok Lab, Seoul National University
