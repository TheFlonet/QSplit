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

## C++ Library used

- [XTensor](https://github.com/xtensor-stack/xtensor) - included in the project (with XTL and XSIMD)
- [XFrame](https://github.com/xtensor-stack/xframe) - included in the project
- [Minorminer](https://github.com/dwavesystems/minorminer/tree/main) - included in the project
- [CPP-Base64](https://github.com/ReneNyffenegger/cpp-base64/tree/master) - included in the project
- [nlohmann JSON](https://github.com/nlohmann/json) - included in the project
- [LibCURL](https://curl.se/libcurl/) - dinamically linked
