# Model 2 — U-Net Brain Tumor Segmentation
## Overview
Model 2 is a U-Net-based image segmentation model designed to identify the tumor region in brain MRI images.
Unlike a classification model that predicts a single class for an image, U-Net produces a pixel-level segmentation mask that identifies the location of the tumor.
The predicted tumor mask is later used by Model 3 for extracting morphological characteristics of the tumor.
Objective:
The main objectives of Model 2 are:
- Detect the tumor region in an MRI image
- Generate a pixel-level binary tumor mask
- Preserve spatial information using U-Net skip connections
- Provide tumor localization information for the morphology-aware Model 3
## Architecture
        ```text   Brain MRI Image
                           │
                           ▼
                    U-Net Encoder
                           │
             ┌─────────────┼─────────────┐
             │             │             │
             ▼             ▼             ▼
          Features      Features      Features
             │             │             │
          MaxPool       MaxPool       MaxPool
             │             │             │
             └─────────────┼─────────────┘
                           │
                           ▼
                       Bottleneck
                           │
                           ▼
                    U-Net Decoder
                           │
                    Skip Connections
                           │
                           ▼
                     Segmentation Map
                           │
                           ▼
                    Sigmoid Activation
                           │
                           ▼
                    Binary Tumor Mask
