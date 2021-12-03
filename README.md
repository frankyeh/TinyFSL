# Tiny FSL
A simplified re-comiled version of the FSL package 

* Download-and-run executives. No installation required
* Tiny (< 100 mb)
* Windows version available
* Improved multi-core supports

Parent package: FMRIB Software Library v6.0 (version 2111)

## License

TinyFSL follows the original FSL license https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Licence 

## Currently Available Programs

- [x] [eddy](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/eddy)

The eddy in TinyFSL have more multi-core support than the original eddy_openmp version. There are four places in eddy_openmp that cannot be parallelized due to free() error. The one in TinyFSL does not have this problem.
  
- [x] [topup](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/topup/TopupUsersGuide)

The topup in TinyFSL is modified to partially supports multi-core. (The one in original FSL does not)

- [x] [applytopup](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/topup/ExampleTopupFollowedByApplytopup)

Same as the original package

- [x] [bet2](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/BET/UserGuide)

Same as the original package

## Installation

No installation needed.
Download the zip file from the release, unzip it, and run

