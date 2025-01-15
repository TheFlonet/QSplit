# QSplit

A hybrid (classical-quantum) solver for QUBO problems developed with the intention of maximising the use of D-Wave's QPU.

## Compared solvers

- [QBSolv](https://github.com/dwavesystems/qbsolv): hybrid solver originally proposed by D-Wave. Open-source.
- [LeapHybridSampler](https://docs.dwavesys.com/docs/latest/doc_leap_hybrid.html): solver currently recommended by D-Wave. Closed-source.
- QSplit: solver developed by us, available in this repository. Open-source.

For this comparison, we explicitly disregard the purely quantum solver [DWaveSampler](https://docs.dwavesys.com/docs/latest/handbook_qpu.html) because:

- The number of currently available qubits does not allow to deal with arbitrarily large problems
- For small problems, QSplit solves directly via QPU

## About the dataset

The problems discussed are a subset of the [QPlib dataset](https://qplib.zib.de/doc.html).

In particular, we dealt the problems with:

- **O** equal to *Q*,
- **V** equal to *B* and,
- no slack variables.

## LaTeX style for the report

[EPTCS](https://github.com/EPTCS/style)
