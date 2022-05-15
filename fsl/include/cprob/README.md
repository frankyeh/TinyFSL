cprob
=====

This is a C++ wrapper around the [CEPHES `cmath`
library](https://netlib.org/cephes/).


In older FSL releases, `cprob` was compiled as a C library and statically
linked into executables. The functions provided by `cprob` were also included
into the `MISCMATHS` C++ namespace.


With the transition to dynamic linking of FSL libraries, managing symbol
resolution is much easier when compiling the `cprob` functions as a C++
library with their own `CPROB` namespace. The `MISCMATHS` namespace hack is
still available by importing the `cprob/libprob.h` header file.
