This repository contains part of the code used in my MSc thesis at I3A about GNNs for free surface fluid simulations. It contains code for a computer vision model used to identify the fluid volume inside a glass.

## Dataset

The dataset contains manually segmented images of the frames of many videos using different glasses and fluids. The first step is to segment the images and preprocess them into a correct .pt format for training.

## Model

After considering different UNet like models, we finally use transfer learning using a pre-trained model (ResNet18) and fine tune it with our dataset. The size of the model and the inference time is critical for a digital twin model.


## Training 

The model is trained in a NVIDIA RTX3090 and many checkpoints are stored to then select the one with the smallest validation loss.


## Inference

At the end the model is used to, in each frame of a real-time video, identify the fluid part. Then, knowing the technical specifications of the camara, we can do real measurements and reconstruct the whole fluid volume. Finally, that volume is meshed for a later particle-based simulation.