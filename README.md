<div id="table-of-contents">
<h2>Table of Contents</h2>
<div id="text-table-of-contents">
<ul>
<li><a href="#sec-1">1. HDPG1D</a></li>
<li><a href="#sec-2">2. Install</a></li>
<li><a href="#sec-3">3. Usage</a></li>
</ul>
</div>
</div>

# HDPG1D<a id="sec-1" name="sec-1"></a>

This is a python package that solves 1-Dimensional PDEs using hybridizable discontinuous Petrov-Galerkin discretization, a novel FEA discretization that ensures the best solution quality without extra stablization mechanism. The following image shows the solution of 1D inviscid Burger's equation using HDPG discretization. Notice that the oscillations near the shock are well controlled even without any artificial stablization mechanism.

<p align="center">
<a href="http://imgur.com/HrWIi4s"><img align="centre" img src="http://i.imgur.com/HrWIi4s.png" width="50%" height="50%" title="source: imgur.com" /></a>
</p>

# Install<a id="sec-2" name="sec-2"></a>
In terminal: 
```bash
pip install hdpg1d
```

# Usage<a id="sec-3" name="sec-3"></a>
In terminal, call:
```bash
PGsolve
```
