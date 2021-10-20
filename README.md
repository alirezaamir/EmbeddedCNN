# Epileptic Seizure Detection - CNN - Fixed point

## Building and running

You can run this application in the PULP and PULPissimo platform with the Makefile 
* [Pulp.mk]

[Pulp]: https://www.pulp-platform.org/
[Pulp.mk]: /Pulp.mk

### PULP

Compile and run with
```sh
make --file=Pulp.mk clean all run
```

### PULPissimo

Compile and run with
```sh
make --file=Pulp.mk clean linker_cp all run
```

## Data files

The input data is in [src/fcn.c](https://c4science.ch/source/C_HW/browse/fcn_pulp/src/fcn.c)