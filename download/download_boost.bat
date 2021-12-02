curl -L https://boostorg.jfrog.io/artifactory/main/release/1.77.0/source/boost_1_77_0.tar.bz2 --output boost.tar.bz2

7z x boost.tar.bz2
7z x boost.tar
del *.tar
del *.tar.bz2

move ./boost_1_77_0/boost ./include
rmdir /S /Q boost_1_77_0

