# GraphSAGE Implementation for SWOW Word Embeddings

## Executive (Claude) Summary

This implementation provides a baseline GraphSAGE encoder with contrastive learning for generating word embeddings from the Small World of Words (SWOW) network. The approach combines graph neural networks with self-supervised learning objectives, using both contrastive learning and skip-gram style losses. The system is designed for the NYT Connections word association game but serves as a general framework for learning semantic representations from lexical networks.

## Architecture Overview

### graphsage_model.py

The core model implements a multi-layer GraphSAGE encoder with the following components:

**GraphSAGEEncoder**: 2-layer GraphSAGE with configurable aggregation (currently mean only), batch normalization, and dropout. Takes pretrained word embeddings as input features and produces 128-dimensional node embeddings.

**ProjectionHead**: MLP that maps embeddings to 64-dimensional contrastive learning space for NT-Xent loss computation.

**GraphSAGEContrastive**: Main model combining encoder and projection head, supporting both embedding extraction and contrastive learning workflows.

**GraphAugmentor**: Implements three augmentation strategies - edge dropout (preferentially removing weak edges), node feature masking, and edge weight jittering for generating contrastive pairs.

**Loss Functions**: NT-Xent contrastive loss and skip-gram negative sampling loss with configurable weighting.

### graphsage_baseline.py

Training pipeline with the following key components:

**GraphSAGETrainer**: Handles model training with support for both neighbor sampling and full-batch modes. Implements weighted combination of contrastive and skip-gram losses with λ=0.7 weighting.

**Random Walk Sampling**: Generates positive samples through graph traversal with edge-weight-based neighbor selection for skip-gram objective.

**Training Loop**: 20 epochs with Adam optimizer (lr=1e-3, weight_decay=1e-5), batch size 512, saving best model based on total loss. Will try 100.

**Evaluation**: Basic similarity testing and connections game validation.

## Current Implementation Status

### Implemented Features

**Graph Processing**: Loads SWOW as directed weighted graph with edge strength filtering (min_strength=0.05). Uses sentence-transformer embeddings as initial node features.

**GraphSAGE Architecture**: 2-layer encoder with mean aggregation, batch normalization, and dropout. Supports neighbor sampling via PyTorch Geometric's NeighborLoader.

**Contrastive Learning**: NT-Xent loss with graph augmentations including edge dropout, node masking, and weight jittering. Temperature scaling (0.1) and projection to 64d space.

**Skip-gram Objective**: Random walk-based positive sampling with negative sampling using uniform node distribution.

**Training Infrastructure**: Combined loss optimization with configurable weighting, model checkpointing, and basic evaluation metrics.

## Missing Components and Limitations

### Graph Data Processing

Edge weight normalization is absent - currently uses raw SWOW strength values without log or power transforms as specified in baseline plan. No integration of psycholinguistic features beyond pretrained embeddings.

### Model Architecture

Only mean aggregation implemented - missing pooling, LSTM, and attention-based aggregators. GraphSAGE layers don't utilize edge weights due to SAGEConv limitations. No stratified sampling by edge strength.

### Training Objectives

Random walk sampling doesn't implement proper weighted sampling P(v|u) ∝ w(u→v)^α. Negative sampling uses uniform distribution rather than unigram^0.75 distribution. Early stopping not implemented.

### Evaluation Framework

No link prediction evaluation with AUC/MRR metrics on held-out edges. Missing word retrieval evaluation with MAP@k scoring. No clustering quality assessment with NMI/ARI against lexical categories. Connections game testing lacks quantitative performance metrics.

### Training Configuration

Only 20 epochs vs recommended 100-300 with early stopping. No learning rate scheduling or advanced optimization strategies.
