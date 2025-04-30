# Optical Flow Analysis and Frame Interpolation

## Project Overview

This project my assignment 3 for Computer Vision course. This project implements and analyzes various optical flow estimation techniques for computer vision applications. The implementation includes multiple approaches to calculate optical flow between image pairs, along with frame interpolation capabilities. The project demonstrates strong understanding of computer vision concepts, Python programming, and image processing techniques.

## Key Features

- Ground truth flow visualization using FlowPy
- Multi-channel optical flow computation using Farnebäck's algorithm
- Comparative analysis of different color channels (RGB, Green, Grayscale)
- Forward and backward flow estimation
- Middle frame interpolation using optical flow
- Lucas-Kanade optical flow implementation
- Comprehensive error analysis using End-Point Error (EPE) metrics

## Technical Implementation

The project is implemented in Python and leverages several key computer vision libraries:

- OpenCV (cv2) for image processing and optical flow computation
- FlowPy for flow visualization and file handling
- NumPy for numerical computations
- Matplotlib for visualization

## Key Components

1. **Ground Truth Visualization (Q1)**

   - Visualizes optical flow ground truth data
   - Supports multiple datasets (Grove3 and RubberWhale)
   - Generates comparative visualizations of frame pairs and flow

2. **Multi-Channel Flow Analysis (Q2)**

   - Implements optical flow computation for individual color channels
   - Provides comparative analysis of different channel performances
   - Calculates and reports End-Point Error (EPE) metrics

3. **Bidirectional Flow Estimation (Q3)**

   - Implements forward and backward flow computation
   - Analyzes flow consistency in both directions
   - Provides error metrics for flow accuracy

4. **Frame Interpolation (Q4)**

   - Estimates intermediate frames using computed optical flow
   - Implements forward and backward warping
   - Calculates RMS error against ground truth middle frames

5. **Advanced Flow Techniques (Q5)**
   - Implements Lucas-Kanade optical flow method
   - Provides alternative flow computation approaches
   - Includes feature tracking and visualization

## Results and Performance

The implementation achieves competitive results on standard optical flow datasets:

- RGB Channel Average EPE: 1.34 (Grove3 dataset)
- Green Channel Average EPE: 1.37 (Grove3 dataset)
- Comparable performance metrics on RubberWhale dataset

## Dependencies

- Python 3.x
- OpenCV
- NumPy
- FlowPy
- Matplotlib

## Project Structure

```
.
├── assign_3_group_09.py    # Main implementation file
├── data/                   # Dataset directory
│   ├── Grove3/            # Grove3 dataset
│   └── RubberWhale/       # RubberWhale dataset
└── output/                 # Generated visualizations and results
```

## Technical Highlights

- Implementation of multiple optical flow algorithms
- Robust error analysis and performance metrics
- Efficient image processing pipeline
- Comprehensive visualization capabilities
- Clean, well-documented code structure
