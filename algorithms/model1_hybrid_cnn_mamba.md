Model 1 — Hybrid CNN-Mamba Brain Tumor Classifier
Overview:
Model 1 is a hybrid deep learning architecture designed for brain MRI tumor classification
The model combines:
- Convolutional Neural Networks (CNNs) for local spatial feature extraction
- Mamba State Space Modeling for modeling dependencies between spatial feature tokens
- A fully connected classification layer for final tumor prediction

The objective is to combine the strong local feature extraction capability of CNNs with the sequence modeling capability of Mamba.

## Architecture

                    Brain MRI Image
                           │
                           ▼
                    Resize 128 × 128
                           │
                           ▼
                 ┌────────────────────┐
                 │   CNN Feature      │
                 │    Extraction      │
                 └─────────┬──────────┘
                           │
                  ┌────────┴────────┐
                  │                 │
                  ▼                 ▼
              Conv2D 3→32      BatchNorm
                  │
                 ReLU
                  │
               MaxPool
                  │
                  ▼
              Conv2D 32→64
                  │
              BatchNorm
                  │
                 ReLU
                  │
                  ▼
          AdaptiveAvgPool 8 × 8
                  │
                  ▼
            64 × 8 × 8 Feature Map
                  │
                  ▼
          Spatial Tokenization
                  │
                  ▼
       64 Spatial Tokens × 64 Features
                  │
                  ▼
             Mamba Block
                  │
                  ▼
              4096 Features
                  │
                  ▼
          Fully Connected Layer
                  │
                  ▼
            Tumor Prediction
