[![Python application](https://github.com/nsarpangala/truss_network_spectral_node/actions/workflows/python-app.yml/badge.svg)](https://github.com/nsarpangala/truss_network_spectral_node/actions/workflows/python-app.yml)

# Network Laplacian Approach to Computing Mechanical Responses of Truss Structures

## Overview

This repository contains code for computing dynamical responses of truss networks. We follow a network Laplacian approach for computing mechanical responses spectrally (Fourier space in time) as explained in the paper (Link).

<div style="display: inline-block; text-align: center;">
<img src="https://github.com/user-attachments/assets/8a1ad73a-148e-400e-861c-da62169d3e1d" alt="Comparison between the lumped mass matrix method (balls and springs) and the truss network method in a disordered network (inset)." width="500"/>
<div style="width: 500px; margin-top: 4px; font-style: italic; font-size: 14px;">
    Figure: Comparison between the  responses from the lumped mass matrix method (balls and springs) and network Laplacian method (this repository) for a disordered truss network with harmonic excitation at one node (inset). It can be seen that beyond a certain frequency ( the Debye frequency ), balls and spring network fails to capture the dynamical responses of networks, whereas our method continues to give a consistent response.
  </div>
</div>


## Features presented in this repository

- **Laplacian Matrix Computation**: Functions to compute the Laplacian matrix for the Truss network-based method we developed and corresponding balls-and-springs models.
- **Impedance Analysis**: Analysis of node responses based on varying impedance values for different rods in the network, as an example is provided in example_problems folder.

## Files

- codes are in src/ folder
- Example Jupyter notebooks and Python script are in example_problems folder

## Installation and Use

Clone the repository
```
git@github.com:nsarpangala/truss_network_spectral_node.git
```

Change directory to the project directory
```
cd truss_network_spectral_node
```

Create a new Python environment and install required libraries.
```
conda create --name test_env python=3.9
conda activate test_env
pip install -r requirements.txt
```

## Testing

To check if everything is working correctly, you can can run pytest

Install pytest library
```
pip install pytest
```
Run pytest on the terminal and ensure the tests are passing successfully
```
pytest
```

## Example use

```
python example_problems/loop_impedance_displacement_boundary.py
```
This will create a folder called `data/Square_Crossbar_Lambda2_constant_mass/` which has a plot of the network `network.png` and the response of the network as a function of the impedance of crossbar, `response.png`. The plots should look as follows.


<img src="https://github.com/user-attachments/assets/37058024-7a33-4067-968d-abde23fa05d4" alt="example image" width="800"/>

## Contributing

Contributions are welcome! Please fork the repository and submit pull requests for any improvements or bug fixes.

## Contact

For any questions or issues, please open an issue on GitHub or contact Niranjan Sarpangala <niras at sas dot upennn dot  edu> or Eleni Katiifori <katifori at sas dot upenn dot edu>

## Contributors
Niranjan Sarpangala, Sean Fancher, Prashant Purohit, Eleni Katifori

University of Pennsylvania, Philadelphia

