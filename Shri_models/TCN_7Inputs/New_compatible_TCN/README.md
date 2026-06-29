# Temporal Convolutional Network (TCN) - Train/Test Array Splitting

This repository utilizes a specific index-masking and segment-extraction approach to partition the trajectory arrays into training and testing sets while preserving sequential dependency.

## Train/Test Splitting Strategy

Because tracking data is sequential and statistical stability is a priority, the data from each input array is split into **Train** and **Test** sets using a structured temporal segment split rather than a random shuffle.

### How the Split is Done:
1. **Per-Array Partitioning:** Each trajectory array is sliced along its primary temporal/sequence axis to prevent future lookahead bias.
2. **Segment Allocation:** For every continuous sequence array:
   * The initial **70%** of the continuous sequence timeline is allocated as the **Training Set** to let the model capture localized velocity gradients.
   * The  **20%** of the timeline is reserved as the **Testing Set** to validate how well the network generalizes to future frames.
   * The remaining **10%** of the timeline is used for  **Validation Set** .
3. **Shape Preservation:** The extraction maintains the 7-channel sequence configuration required by the network layers in `base_tcn_7in.pth`, guaranteeing uniform array shapes across both sets after the split.
