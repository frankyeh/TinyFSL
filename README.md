# Tiny FSL
A simplified patched recompiled version of the FSL package 

<a href="https://github.com/frankyeh/TinyFSL/commits/master"><img src="https://img.shields.io/github/last-commit/frankyeh/TinyFSL"><a href="https://github.com/frankyeh/TinyFSL/releases"><img src="https://img.shields.io/github/v/release/frankyeh/TinyFSL"></a><a href="https://github.com/frankyeh/TinyFSL/releases"><img src="https://img.shields.io/github/downloads/frankyeh/TinyFSL/total?style=social"></a>
  
* Available in Windows version
* Download-and-run. No installation required
* ~20 MB 
* Improved multi-core supports

## License

TinyFSL follows the original FSL license https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Licence 
 
Parent package: FMRIB Software Library v6.0 (version 2111)

## Currently Available Programs

- [x] [eddy](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/eddy)

The eddy in TinyFSL has more multi-core support than the original eddy_openmp version. There are four places in the original eddy_openmp version that cannot be parallelized due to free() error. The eddy in TinyFSL solved this limitation and was potentially faster than eddy_openmp.
  
- [x] [topup](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/topup/TopupUsersGuide)

The topup in TinyFSL is modified to support multi-core partially. (The one in original FSL does not).
Mot topup calculation is sequential, but some computation bottlenecks can be handled by parallel processing.

- [x] [applytopup](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/topup/ExampleTopupFollowedByApplytopup)

Same as the original package

- [x] [bet2](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/BET/UserGuide)

Same as the original package

- [ ] more will be added...

## Installation

No installation is needed.
Download the zip file from the release, unzip it, and run

