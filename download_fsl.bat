curl https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/linux-64/fsl-armawrap-0.6.0-h2bc3f7f_0.tar.bz2 --output fsl-armawrap-0.6.0-h2bc3f7f_0.tar.bz2
curl https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/linux-64/fsl-avwutils-2111.0-h2bc3f7f_0.tar.bz2 --output fsl-avwutils-2111.0-h2bc3f7f_0.tar.bz2
curl https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/linux-64/fsl-bet2-2111.0-h2bc3f7f_0.tar.bz2 --output fsl-bet2-2111.0-h2bc3f7f_0.tar.bz2
curl https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/linux-64/fsl-basisfield-2111.0-h2bc3f7f_0.tar.bz2 --output fsl-basisfield-2111.0-h2bc3f7f_0.tar.bz2
curl https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/linux-64/fsl-cprob-2111.0-h2bc3f7f_0.tar.bz2 --output fsl-cprob-2111.0-h2bc3f7f_0.tar.bz2
curl https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/linux-64/fsl-eddy-2111.0-h2bc3f7f_0.tar.bz2 --output fsl-eddy-2111.0-h2bc3f7f_0.tar.bz2 
curl https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/linux-64/fsl-meshclass-2111.0-h2bc3f7f_0.tar.bz2 --output fsl-meshclass-2111.0-h2bc3f7f_0.tar.bz2
curl https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/linux-64/fsl-miscmaths-2111.1-h2bc3f7f_0.tar.bz2 --output fsl-miscmaths-2111.1-h2bc3f7f_0.tar.bz2
curl https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/linux-64/fsl-newimage-2111.0-h2bc3f7f_0.tar.bz2 --output fsl-newimage-2111.0-h2bc3f7f_0.tar.bz2
curl https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/linux-64/fsl-newimage-2111.0-h2bc3f7f_0.tar.bz2 --output fsl-newimage-2111.0-h2bc3f7f_0.tar.bz2
curl https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/linux-64/fsl-newnifti-4.0.0-h2bc3f7f_0.tar.bz2 --output fsl-newnifti-4.0.0-h2bc3f7f_0.tar.bz2 
curl https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/linux-64/fsl-topup-2111.0-h2bc3f7f_0.tar.bz2 --output fsl-topup-2111.0-h2bc3f7f_0.tar.bz2
curl https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/linux-64/fsl-utils-2111.1-h2bc3f7f_0.tar.bz2 --output fsl-utils-2111.1-h2bc3f7f_0.tar.bz2
curl https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/linux-64/fsl-warpfns-2111.0-h2bc3f7f_0.tar.bz2 --output fsl-warpfns-2111.0-h2bc3f7f_0.tar.bz2
curl https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/linux-64/fsl-znzlib-2111.0-h2bc3f7f_0.tar.bz2 --output fsl-znzlib-2111.0-h2bc3f7f_0.tar.bz2
move *.bz2 fsl
cd fsl
7z e *.bz2
del *.bz2
7z x *.tar -y
del *.tar