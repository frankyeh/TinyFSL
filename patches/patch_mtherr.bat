call powershell -Command "(gc include\cprob\mtherr.cc) -replace '\(char ', '(const char ' | Out-File -encoding ASCII include\cprob\mtherr.cc"
call powershell -Command "(gc include\cprob\mtherr.cc) -replace '\( char ', '(const char ' | Out-File -encoding ASCII include\cprob\mtherr.cc"

call powershell -Command "(gc include\cprob\mtherr.cc) -replace 'static char ', 'static const char ' | Out-File -encoding ASCII include\cprob\mtherr.cc"

call powershell -Command "(gc include\cprob\mconf.h) -replace '\( char ', '(const char ' | Out-File -encoding ASCII include\cprob\mconf.h"

call powershell -Command "(gc include\cprob\cprob.h) -replace '\(char ', '(const char ' | Out-File -encoding ASCII include\cprob\cprob.h"
call powershell -Command "(gc include\cprob\cprob.h) -replace '\( char ', '(const char ' | Out-File -encoding ASCII include\cprob\cprob.h"

