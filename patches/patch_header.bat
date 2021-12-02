


powershell -Command "(gc include\newimage\imfft.h) -replace '#include <unistd.h>', '' | Out-File -encoding ASCII include\newimage\imfft.h"
powershell -Command "(gc include\utils\FSLProfiler.h) -replace '#include <sys/time.h>', '' | Out-File -encoding ASCII include\utils\FSLProfiler.h"
powershell -Command "(gc include\eddy\EddyHelperClasses.h) -replace '#include <sys/time.h>', '' | Out-File -encoding ASCII include\eddy\EddyHelperClasses.h"


