```python

```

# QDataSet: Quantum Datasets for Machine Learning

## Overview 
The QDataSet comprises 52 datasets based on simulations of one- and two-qubit systems evolving in the presence of absence of noise subject to a variety of controls. It has been developed to provide a large-scale set of datasets for the training, benchmarking and competitive development of classical and quantum algorithms for common tasks in quantum sciences, including quantum control, quantum tomography and noise spectroscopy. It has been generated using customised code drawing upon base-level Python packages in order to facilitate interoperability and portability across common machine learning and quantum programming platforms. Each dataset consists of 10,000 samples which in turn comprise a range of data relevant to the training of machine learning algorithms for solving optimisation problems. The data includes a range of information (stored in list, matrix or tensor format) regarding quantum systems and their evolution, such as: quantum state vectors, drift and control Hamiltonians and unitaries, Pauli measurement distributions, time series data, pulse sequence data for square and Gaussian pulses and noise and distortion data. 

The total compressed size of the QDataSet (using Pickle and zip formats) is around 14TB (uncompressed, around 100TB). The datasets are stored on UTS Cloudstor cluster. Researchers can use the QDataSet in a variety of ways to design algorithms for solving problems in quantum control, quantum tomography and quantum circuit synthesis, together with algorithms focused on classifying or simulating such data. [We also provide working examples of how to use the QDataSet in practice and its use in benchmarking certain algorithms.] Each part below provides in-depth detail on the QDataSet for researchers who may be unfamiliar with quantum computing, together with specifications for domain experts within quantum engineering, quantum computation and quantum machine learning.
 
As discussed above, the aim of generating the datasets is threefold: (a) simulating typical quantum engineering systems, dynamics and controls used in laboratories; (b) using such datasets as a basis to train machine learning algorithms to solve certain problems or achieve certain objectives, such as attainment of a quantum state $\rho$, quantum circuit <img src="https://render.githubusercontent.com/render/math?math=%24%5CPi_i%20U_i%24"> or quantum control problem generally (among others); and (c) enable optimisation of algorithms and spur development of optimised algorithms for solving problems in quantum information, analogously with the role of large datasets in the classical setting.

# Description of datasets

The datasets in the QDataSet are set-out in Pickle-compressed lists and dictionaries. A taxonomy of each datasets is included below.

## QDataSet Structure

Each datatset in the QDataSet consists of 10,000 examples. An example corresponds to a given control pulse sequence, associated with a set of noise realizations. Every dataset is stored as a compressed zip file, consisting of a number of Python \textit{Pickle} files that stores the information. Each file is essentially a dictionary consisting of the elements described in the paper. The datasets were generated on the University of Technology (Sydney) high-performance computing cluster (iHPC) [\hl{\textbf{ref}}]. The QDataSet was generated on using the iHPC Mars node (one of 30). The node consists of Intel Xeon Gold 6238R 2.2GHz 28cores (26 cores enabled) 38.5MB L3 Cache (Max Turbo Freq. 4.0GHz, Min 3.0GHz) 360GB RAM. We utilised GPU resources using a NVIDIA Quadro RTX 6000 Passive (3072 Cores, 384 Tensor Cores, 16GB Memory). It took around three months to generate over 2020-2021, coming to around 14TB of compressed quantum data. Single-qubit examples were relatively quick (between a few days and a week or so). The two-qubit examples took much longer, often several weeks.

#### 

| Item                     | Description                                                                                                                                                                                                                                                                                              |
|---------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| simulation parameters |                                                                                                                                                     * dim: the dimension $2^n$ of the Hilbert space for $n$ qubits (dimension 2 for single qubit, 4 for two qubits);      * $\Omega$: the spectral energy gap;
     * static operators: a list of matrices representing the time-independent parts of the Hamiltonian (i.e. drift components);
     * dynamic operators: a list of matrices representing the time-dependent parts of the Hamiltonian (i.e. control components), without the pulses. So, if we have a term $f(t) \sigma_x + g(t) \sigma_y$, this list will be $[\sigma_x, \sigma_y]$;
     * noise operators}: a list of time-dependent parts of the Hamiltonian that are stochastic (i.e. noise components). so if we have terms like $\beta_1(t) \sigma_z + \beta_2(t) \sigma_y$, the list will be $[\sigma_z, \sigma_y]$;
     * measurement operators}: Pauli operators (including identity) ($I,\sigma_x,\sigma_y, \sigma_z$)
     * initial states: the six eigenstates of the Pauli operators;
     * T}: total time (normalised to unity);
     * num ex}: number of examples, set to 10,000;
     * batch size: size of batch used in data generation (default is 50);
     * $K$: number of randomised pulse sequences in Monte Carlo simulation of noise (set to $K = 2000$);
     * noise profile: N0 to N6 (see above);
     * pulse shape: Gaussian or Square;
     * num pulses: number of pulses per interval;
     * elapsed time: time taken to generate the datasets.                                                                                                                                                    |
| pulse parameters      | The control pulse sequence parameters for the example:                                                                                                                                                                                                                                                            |
| time range            | A sequence of time intervals $\Delta t_j$ with $j = 1,...,M$                                                                                                                                                                                                                                                      |
| pulses                 | Time-domain waveform of the control pulse sequence.                                                                                                                                                                                                                                                               |
| distorted pulses      | Time-domain waveform of the distorted control pulse sequence (if there are no distortions, the waveform will be identical to the undistorted pulses).                                                                                                                                                             |
| expectations           | The Pauli expectation values 18 or 52 depending on whether one or two qubits (see above). For each state, the order of measurement is: $\sigma_x, \sigma_y, \sigma_z $ applied to the evolved initial states. As the quantum state is evolving in time, the expectations will range within the interval $[1,-1]$. |
| $V_O$ operator         | The $V_O$ operators corresponding to the three Pauli observables, obtained by averaging the operators $W_O$ over all noise realizations.                                                                                                                                                                          |
| noise                  | Time domain realisations of the relevant noise.                                                                                                                                                                                                                                                                   |
| $H_0$                  | The system Hamiltonian $H_0(t)$ for time-step $j$.                                                                                                                                                                                                                                                                |
| $H1$                   | The noise Hamiltonian $H_1(t)$ for each noise realization at time-step $j$.                                                                                                                                                                                                                                       |
| $U_0$                  | The system evolution matrix $U_0(t)$ in the absence of noise at time-step $j$.                                                                                                                                                                                                                                    |
| $U_I$                  | The interaction unitary $U_I(t)$ for each noise realization at time-step $j$.                                                                                                                                                                                                                                     |
| $V_O$                  | Set of $3 \times 2000$ expectation values (measurements) of the three Pauli observables for all possible states for each noise realization. For each state, the order of measurement is: $\sigma_x, \sigma_y, \sigma_z $ applied to the evolved initial states.                                                   |
| $E_O$                  | The expectations values (measurements) of the three Pauli observables for all possible states averaged over all noise realizations. For each state, the order of measurement is: $\sigma_x, \sigma_y, \sigma_z $ applied to the evolved initial states.                                                           |
                                                          |


```
