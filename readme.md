# Project Name

This repository is a practical example of an issue regarding backpropagating through a Mujco-mjx physics step.
The original purpose was to find out whether we are able to backpropagate through an mjx step via a minimal example.
We initialize a state ```c0``` in the of an environment and add a randomized $\epsilon$ to its position and velocity which results in ```v0```.
We now advance a mjx_step with ```ctrl=0```, resulting in ```c1``` and ```v1```. The computed loss is the squared difference of the position and velocity difference of ```c1``` and ```v1```. Via gradient-descent we are trying to change ```v0``` such that ```v1``` gets closer to ```c1```. Which should ultimately result in ```v0=c0```.


## Table of Contents

- [Installation](#installation)
- [Usage](#usage)


## Installation

Instructions on how to install and set up your project.

## Usage

The main code is provided in the Backpropagation_test notebook. While the environment is described in a xml file and then loaded in qube_mjx.py

