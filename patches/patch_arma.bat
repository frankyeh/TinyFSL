


powershell -Command "(gc include\newimage\generalio.cc) -replace 'BUILDSTRING', '\"WIN_FSL\"' | Out-File -encoding ASCII include\newimage\generalio.cc"
