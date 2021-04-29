<img height='60px' src='doc/logo/msu.gif'/>

## Soft Deep Q-Learning with cross entropy regulation. 
This repository contains multiple deep Q-learning algorithm with an emphasis on faster and better convergence. A newton raphson approximation is explored in the update rule and compared against a vanilla deep Q-network (DQN) and against a novel algorithm named Soft-Q CE, which augment a novel cross entropy loss function.

## Description
In the search for faster and better convergence of Deep Q-Learning methods we augment the original temporal difference (TD) loss with a novel loss function, i.e., with cross entropy (CE) loss. We use Double Q-Learning method consisting of a main and a target network. The CE loss is calculated between the main and the target network.

<p align="center">
<img src="doc/CE_loss.png" width="300">
</p>

The augment CE loss is compared against a benchmark DQN. Additionally, we have explored a newton raphson method approximation for updating the weight of the DQN extened from TDProp algorithm (check reference).  
In the repository we present three results: 
- Study of depth and width of neural network used in DQN.
- Performance evaluation using TDProp Q.
- Performance evalutaion using soft-Q CE and comparison against other methods.

## Installation

### Prerequisites

Requires python 3 (>=3.5). You'll also need basic python packages pip, numpy and pickle.
PyTorch (CUDA optional) is required for TDProp Q and Soft-Q CE algorithm. The repository is currently tested with PyTorch
v1.8.1 with designed compatibility to older versions.

Alternate vanilla DQN uses tensorflow and keras v.

### Installing

- Clone the repo and cd into it:
    ```bash
       git clone https://github.com/ECE884Group1/ECE884_Final_Project.git
       cd ECE884_Final_Project
    ```
- If you don't have Pytorch installed already, install your favourite flavor of Pytorch. Please visit the following for [installation](https://pytorch.org/get-started).
- If you don't have gym installed already. In most cases, you may use
  ```bash
    pip install gym
    ```
    
- If you don't have TensorFlow installed already, install your favourite flavor of TensorFlow. In most cases, you may use
    ```bash 
    pip install tensorflow-gpu==1.14 # if you have a CUDA-compatible gpu and proper drivers
    ```
    or 
    ```bash
    pip install tensorflow==1.14
    ```
    to install Tensorflow 1.14, which is the latest version of Tensorflow supported by the master branch. Refer to [TensorFlow installation guide](https://www.tensorflow.org/install/)
    for more details. 
    

## Usage 


## Results
We evaluate the developed algorithms in the CartPole v1 and MountainCar v0.

<p float="center">
  <img src="doc/MountainCar-v0.gif" width="300" />
  <img src="doc/CartPole-v1.gif" width="300" />
</p>

The convergence plot for the TDProp Q and Soft-Q CE for the MountainCar environment are shown below 
<p float="center">
  <img src="doc/results/DQN CE vs D r_train (MC).png" width="300" />
  <img src="doc/results/DQN TD vs D r_train (MC).png" width="300" />
</p>
The convergence plot for the TDProp Q and Soft-Q CE for the CartPole environment are shown below 
<p float="center">
  <img src="doc/results/DQN CE vs D r_train (CP).png" width="300" />
  <img src="doc/results/DQN TD vs D r_train (CP).png" width="300" />
</p>

|      Algorithm     | Average rewards Cart Pole | Average rewards Mountain Car |
|:------------------:|:-------------------------:|:----------------------------:|
| 2 Hidden layer DQN |           206.4           |            -118.21           |
| 4 Hidden layer DQN |            500            |            -101.07           |
| 7 Hidden layer DQN |           106.45          |            -107.07           |
|     Deep 96 DQN    |            500            |            -101.60           |
|    Deep 128 DQN    |            500            |            -126.47           |
|    Deep 256 DQN    |            500            |            -107.66           |
|     TDProp DQN     |            500            |            -121.02           |
|      Soft-Q CE     |            500            |            -100.01           |

## Citation

```
@article{Soft_Q_CE,
      title={Soft Q-Learning with cross entropy regulation}, 
      author={Sandeep Banike, Hrishikesh Dutta, Amit Kumar Bhuyan and Avirup Roy},
      year={2021},
      journal = {GitHub repository},
  	  howpublished = {\url{https://github.com/ECE884Group1/ECE884_Final_Project}
}
```

## Troubleshootings / Discussion

If you have any problem using repository or want to be a part of the project, contact any of the authors.
