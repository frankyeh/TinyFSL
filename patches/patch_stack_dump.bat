powershell -Command "(gc include\eddy\eddy.cpp) -replace '#include \"utils/stack_dump.h\"', '' | Out-File -encoding ASCII include\eddy\eddy.cpp"
powershell -Command "(gc include\eddy\eddy.cpp) -replace 'StackDump::Install\(\);', '' | Out-File -encoding ASCII include\eddy\eddy.cpp"


powershell -Command "(gc include\topup\topup.cpp) -replace '#include \"utils/stack_dump.h\"', '' | Out-File -encoding ASCII include\topup\topup.cpp"
powershell -Command "(gc include\topup\topup.cpp) -replace 'StackDump::Install\(\);', '' | Out-File -encoding ASCII include\topup\topup.cpp"