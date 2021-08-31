# Three State Cyclic Memory Model

Is it possible to manipulate memories? Memory is a process of storing and retrieving information by the brain cells known as neurons. If there are damages to the neurons such that they cannot transfer signals between them, you can start to experience difficulties remembering familiar tasks.   

We want to understand memory formation on a microscale and find universal rules that govern information maintenance length.  

## Introduction

The inspiration for this project came from a paper wrote by professor Dante Chialvo et al., titled Noise-induced memory in extended excitable systems.  

At the neural level, memory is defined as the activation of neurons from voltage stimuli.
In this project, we replicated such rules in a computational model of a one-dimensional square lattice. Each site on the lattice represents a neuron, and it can take one of the 
three states:
```
Quiescent ------> Excited ------> Refractory ------> ...
 
```


## Objectives

The aim of this project is to reproduce everything in the original paper and explore other initial conditions.  
The 1D system has open and closed boundary conditions imposed, and the maximum separation of two nodes differ in each case  
- Open BC: maximum separation is the size of the lattice
- Closed BC: maximum separation is half the size of the lattice

The memory effect induced by initial node is both evident by eye and the statistical constants.

## Further reading
To fully understand the project outcomes, please read on Greenberg Hastings Cellular Automata and Hurst constant.


