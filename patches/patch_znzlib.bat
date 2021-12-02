(echo #include "zlib.h") > include\znzlib\znzlib.h_
powershell -Command "(gc include\znzlib\znzlib.h) -replace '#include \"zlib.h\"', '' | Out-File -encoding ASCII include\znzlib\znzlib.h2"
type include\znzlib\znzlib.h_ include\znzlib\znzlib.h2 > include\znzlib\znzlib.h
del include\znzlib\znzlib.h2 include\znzlib\znzlib.h_


