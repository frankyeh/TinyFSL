cd fsl
for /f "delims=" %%x in ('dir include\* /b ') do (move /y src\fsl-%%x\* include\%%x) 
rmdir /S /Q bin
rmdir /S /Q info
rmdir /S /Q src
rmdir /S /Q tcl
cd ..
for /f "delims=" %%x in ('dir patches\* /b ') do (move /y patches\%%x\* fsl\include\%%x) 