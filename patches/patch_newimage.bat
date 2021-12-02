powershell -Command "(gc include\NewNifti\nifti2.h) -replace '__attribute__ \(\(__packed__\)\)', '' | Out-File -encoding ASCII include\NewNifti\nifti2.h"

move /y patches\newimage\* include\newimage