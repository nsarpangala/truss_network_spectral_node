[![Python application](https://github.com/nsarpangala/truss_network_spectral_node/actions/workflows/python-app.yml/badge.svg)](https://github.com/nsarpangala/truss_network_spectral_node/actions/workflows/python-app.yml)

# Repository Description

## Overview

This repository contains code for simulating and analyzing mechanical networks, specifically focusing on the dynamics of joints in Fourier space. We follow a graph Laplacian approach for computing mechanical responses.

<div style="display: inline-block; text-align: center;">
<img src="https://github.com/user-attachments/assets/8a1ad73a-148e-400e-861c-da62169d3e1d" alt="Comparison between the lumped mass matrix method (balls and springs) and the truss network method in a disordered network (inset)." width="500"/>
<div style="width: 500px; margin-top: 4px; font-style: italic; font-size: 14px;">
    Figure: Comparison between the  responses from the lumped mass matrix method (balls and springs) and network Laplacian method (this repository) for a disordered truss network (inset).
  </div>
</div>


## Features

- **Laplacian Matrix Computation**: Functions to compute the Laplacian matrix for the Truss network based method we developed and corresponding balls-and-springs models.
- **Impedance Analysis**: Analysis of node responses based on varying impedance values for different rods in the network, as an example is provided in example_problems folder.
- **Boundary Conditions**: Handling of fixed coordinates and boundary conditions in the network.

## Files

- codes are in src/ folder
- example jupyter notebooks and python script are in example_problems folder


## Contributing

Contributions are welcome! Please fork the repository and submit pull requests for any improvements or bug fixes.

## Contact

For any questions or issues, please open an issue on GitHub or contact Niranjan Sarpangala <niras at sas dot upennn dot  edu> or Eleni Katiifori <katifori at sas dot upenn dot edu>

## Contributors
Niranjan Sarpangala, Sean Fancher, Prashant Purohit, Eleni Katifori

Univerity of Pennsylvania, Philadelphia

