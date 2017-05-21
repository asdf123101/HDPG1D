# HDPG1D
This is a python package that solves 1-Dimensional PDEs using hybridizable discontinuous Petrov-Galerkin discretization, a novel FEA discretization that ensures the best solution quality without extra stablization mechanism. The following image shows the solution of 1D inviscid Burger's equation using HDPG discretization. 
```math
\frac{\partial u}{\partial t} + \frac{1}{2} \frac{\partial u^2}{\partial x} = 0, \quad \text{in } \Omega \in [0,1].
```
Notice that the oscillations near the shock are well controlled even without any artificial stablization mechanism.

<p align="center">
<img align="centre" img src="http://i.imgur.com/HrWIi4s.png" width="50%" height="50%" title="source: imgur.com" />
</p>

The main task of this project is to build an automated CFD framework based on HPDG discretization and robust adaptive mesh refinement. The solver is currently capable of solving 1D linear convection-diffusion equations of the following form with Dirichlet boundary condition,
```math
c_1\frac{\partial u}{\partial x} + c_2\frac{\partial^2 u}{\partial x^2} + c_3u = f,
```
where $`c_1`$, $`c_2`$, and $`c_3`$ are constants and $f$ is the forcing term.

## Install
In the souce directory: 
```bash
python setup.py sdist
cd dist/
pip install hdpg1d
```

## Usage
In terminal, call:
```bash
PGsolve
```
Follow the prompts to setup the problem, visualize the solution, and check the convergence.

### Convergence plot
The error on the convergence plot is calculated using approximated 'exact' solution with higher polynomial functions and large number of elements. Therefore, the error plot, especially the adaptive solution error may not be accurate after certain threshold.

### Adaptive method
By default, the solver seeks to adapt the mesh based on the right boundary flux. The solver stops after the estimated error is lower than a user defined tolerance or reaches maximum iteration number.

## Problem setup
`PGsolve` provides two methods to setup the problem:
* Command line interface: when `PGsolve` is called from command line, it offers the option to setup the problem manually instead of the default parameters. The parameters could be changed in command line are
	- convection coefficient $c_1$
	- diffusion coefficient $c_2$
	- reaction coefficient $c_3$
	- order of polynomial basis functions
	- initial number of elements
	- stablization parameters $\tau^+$ and $\tau^-$
* Configuration file: [config.json](config/config.json) is the sample configuration file comes with the application, specifying the default parameters in the command line interface. To customize parameters including forcing term and boundary condtions, create `config.json` in the current working directory or in your home directory `~/`, copy the content of the sample configuration file, and change the value of each entry according to the specific problem.  Running `PGsolve` in the currecnt working directory will read the new configuration file and use the values in the file as default parameters.

## To-do
* Support non-linear problems
* Support time-marching
* More adaptive target functions
* Adaptive HDPG solution routine
