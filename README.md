# Tiny FSL
A simplified redistribution of FSL package

Parent package: FMRIB Software Library v6.0 (version 2111)

## License
TinyFSL follows the original FSL license https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Licence given non-commercial usage.

## Currently Available Programs

- [x] [eddy](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/eddy)

The eddy in TinyFSL have more multi-core support than the original eddy_openmp version. There are four places in eddy_openmp that cannot be parallelized due to free() error. The one in TidyFSL does not have this problem.
  
- [x] [topup](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/topup/TopupUsersGuide)

The topup in TidyFSL is modified to partially supports multi-core. (The one in original FSL does not)

- [x] [applytopup](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/topup/ExampleTopupFollowedByApplytopup)

Same as the original package

- [x] [bet2](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/BET/UserGuide)

Same as the original package

## Installation

No installation needed.
Download the zip file from the release, unzip it, and run

