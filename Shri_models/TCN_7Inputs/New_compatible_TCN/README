# Temporal Convolutional Network (TCN) - Transition to Velocity Dynamics

This repository contains the weights and configuration for the 7-input Temporal Convolutional Network (`base_tcn_7in.pth`)[cite: 4]. This README details the critical workflow modification applied to the tracking sequences to establish statistical stability and ensure robust model convergence[cite: 4].

## Core Methodology: Shift to Velocity Dynamics

### The Problem with Raw Trajectories
Processing sequence-based tracking using absolute, raw spatial coordinates creates severe statistical instabilities[cite: 4]. Models operating directly on absolute spatial domains are highly susceptible to:
* **Tracking Drift:** Accumulated spatial deviations across extended frames[cite: 4].
* **Lack of Translation Invariance:** The model bounds itself to the coordinate grid layout of the training field instead of generalizing movement features[cite: 4].
* **Unbounded Scale Variance:** Extreme variances across different coordinate dimensions destabilizing gradient updates[cite: 4].

### The Solution: Velocity-Based Splitting
To fix this, the input pipeline applies a temporal first-derivative transformation across the coordinate sequence[cite: 4]. Instead of passing absolute grid positions, the trajectory is split into continuous frame-to-frame displacement components[cite: 4]:

$$\Delta x_t = x_t - x_{t-1}$$
$$\Delta y_t = y_t - y_{t-1}$$

This process extracts local, continuous velocity components across the tracking timelines, fundamentally shifting the feature domain from spatial positions to sequential velocity vectors[cite: 4].

## Key Statistical Impacts

1. **Translation Invariance:** By stripping absolute grid contexts, the network evaluates trajectory geometries universally, regardless of where on the coordinate map the sequence began[cite: 4].
2. **Stationary Mean:** Converting the features into velocity dynamics centers the sequence inputs around a stable, consistent mean, preventing network drift[cite: 4].
3. **Bounded Scale:** Velocity components drastically reduce raw scalar variance, eliminating extreme gradient spikes and ensuring steady parameter optimization[cite: 4].

## Model Alignment
The `base_tcn_7in.pth` checkpoint is fully optimized for this velocity-driven paradigm[cite: 4]. The 7-channel input dimension natively digests these multi-component delta trajectories, enabling stable temporal tracking and highly generalized trajectory mapping[cite: 4].
