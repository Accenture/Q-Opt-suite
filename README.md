# Q-Opt Benchmarking Suite

This is a benchmarking suite to used test optimisers, predominantly Quantum and Quantum-inspired ones. The purpose is to test a wide range of optimisation solutions against a varied set of given problems, such as the Traveling Salesman or Feature Selection, in order to get a feeling for the strengths, weaknesses, scaling behaviour and limitations of each platform.

## Introduction

Quantum annealing is a type of quantum algorithm that can be used to solve optimization problems. In contrast to circuit-based quantum computers, it is fixed purpose and has a greater level of maturity as a technology. It works by starting with a system in a superposition of all possible solutions to the problem. The system is then cooled down, and as it does, it tends to collapse into the ground state, which is the lowest energy state. The ground state of the system corresponds to the optimal solution to the problem. The most well-known manufacturer of quantum annealers is [D-Wave](https://www.dwavesys.com/).

Quantum inspired optimization is a field of research that uses ideas from quantum annealing to develop new optimization algorithms that can be implemented on classical computers. These algorithms can often solve problems that are too difficult for classical computers to solve using traditional methods. Quantum-inspired optimisers typically use CPUs, graphics cards, or FPGAs rather than quantum circuits.

Some of the potential applications of quantum inspired optimization using annealers include:

- Scheduling. Scheduling problems involve finding the optimal way to assign tasks to resources. This can be a difficult problem, especially when there are many tasks and resources involved. Quantum inspired optimization algorithms can be used to find better solutions to scheduling problems.
- Routing. Routing problems involve finding the shortest path between two points. This can be a difficult problem, especially when there are many obstacles or constraints. Quantum inspired optimization algorithms can be used to find better solutions to routing problems.
- Logistics. Logistics problems involve planning and managing the movement of goods and materials. This can be a complex problem, especially when there are many different factors to consider. Quantum inspired optimization algorithms can be used to find better solutions to logistics problems.
- Finance. Financial problems involve making decisions about investments, trading, and risk management. These problems can be very complex, and quantum inspired optimization algorithms can be used to find better solutions.

Quantum inspired optimization is a rapidly developing field, and there is a lot of potential for this technology to be used to solve real-world problems. However, it is important to note that quantum annealers are still in their early stages of development, and they are not yet capable of solving all types of optimization problems.

## Running

Run `main.py` with an optional `-f <configuration file>` argument. By default, `default.yml` is used.

The benchmark configurations are specified in the YAML configuration file. Please see benchmarks.yml for an illustration of its structure.

- At the top level, the `models` section provides a default set of models to run at different sizes.
- However each platform can be supplied with its own set of `models` which will override the top-level specification.
- Each hardware platform typically has platform-specific parameters to set.
- At each level -- top, platform, and model -- it is possible to set an `enabled` boolean flag. Lower levels will override the higher-level settings. This allows for easy activation and deactivation of tests or groups of tests.

The results of the benchmarks are written to a results/<datetime> directory in the form of a logfile and a CSV file with the benchmark outcomes. A fixed RNG seed is used for consistency of outcomes.

## Architecture

The goal is to have a set of _M_ optimisation models representing a wide range of problems, and running each of these models against a set of _N_ hardware platforms, at a range of different sizes to gain an idea of how each platform scales.

Subclasses of the `models.Model` class represent an optimisation models. The framework comes with a range of pre-built models:

- **Bin packing** is the naive implementation of the standard problem of packing _N_ items with various weights in _M_ containers all with the same given maximum weight. The optimisation goal is to minimise the number of containers used; because of the discontinuous nature this is a very difficult Hamiltonian landscape for most annealers.
- **Bin packing SP** is a more annealer-friendly two-step solver for the bin packing problem. First, a large number of reasonable packings for the first container is generated, making use of the fact that annealers typically generate many solutions. Step two is an attempt to use this packing set to cover all the items to be packed, in other words, the bin packing problem has reduced to a graph covering problem. Many thanks to Yaz Izumi at Toshiba for his help with this.
- **Bin packing 3D** is a three-dimensional version of the bin packing problem focusing on sizes rather than weights.
- **Feature selection** is inspired by AI, where the selection of the _N_ most relevant of _M_ possible features is an optimisation problem during the training phase. It is a problem with a dense interaction matrix and a single constraint which will test especially hardware platforms with limited connectivity.
- **Max-Cut** is a straightforward encoding of the familiar maximum cut graph problem, representative of a wide range of optimsation problems such as circuit design, data clustering or network flow.
- **Pizza parlour** is a linear programming problem. This problem is not natural territory for annealers as it can be readily solved algorithmically, however, the purpose here is not to provide the most efficient implementation but rather to assess how solver platforms that do not natively support integers behave when confronted with the complex Hamiltonian landscape of binary-encoded integers.
- **Shannon-Kirkpatrick** (SK) is a spin glass model. It has a completely dense interaction matrix and no constraints, and will exercise both connectivity and ability to efficiently find the global optimum in a complex landscape.
- **Traveling Salesman Problem** (TSP) is another standard problem representing a wide range of real-world optimisation problems in areas such as logistics and robot pathing.

The internal representation used is a Qiskit `QuadraticProgram`. It is important to appreaciate that this means the problem isn't necessarily coded down to a QUBO -- if a solver platform natively supports constraints, for example, or supports integers, then the framework will take advantage of that. The purpose is to test each solution at its strongest.

Subclasses of the `platforms.Platform` class represent solver hardware (and/or software) platforms. This framework comes with a range of platforms:

- AWS Braket platforms
  - **[Toshiba SQBM+](https://www.global.toshiba/ww/products-solutions/ai-iot/sbm.html)** a quantum-inspired annealer which works exceptionally well.
- MS Azure platforms ([DEPRECATED by Microsoft](https://learn.microsoft.com/en-us/azure/quantum/optimization-deprecation-warning))
  - **MS Quantum Inspired Optimisation** (MSQIO) (DEPRECATED) Microsoft is shuttering its optimisers, focusing on circuit-based quantum computing instead.
  - **1QBit** (DEPRECATED) Microsoft is shuttering its optimisers, focusing on circuit-based quantum computing instead.
- [D-Wave](https://www.dwavesys.com/) solvers
  - **Binary Quadratic Model** hybrid cloud solver. Restricted to QUBOs; in most cases the CQM solver is superior.
  - **Constrained Quadratic Model** hybrid cloud solver. Performs exceptionally well but the bulk of the work seems to be done by black-box classical algorithms.
  - **Native QPU** solver which runs the problem directly on the QPU. Because of connectivity limitations, the automated embedding quickly runs into trouble on highly connected models.
- Other platforms
  - **[Quantagonia](https://www.quantagonia.com/)** is a quantum(-inspired) annealer and hybrid solver startup.
  - **[Gurobi](https://www.gurobi.com/)** is included as a best-in-class algorithmic solver for comparison purposes.
- Manual operations
  - **Matrix Market Writer** writes the problem as a QUBO in MatrixMarket format in order to run benchmarks outside of the framework.
  - **CSV Reader** correspondingly will read the optimised solutions from CSV format files.

## Extension

In order to add a new model or platform, all you should need to do is add a `Model` or `Platform` subclass. Class names are specified in the configuration file.