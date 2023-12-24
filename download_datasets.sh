
# required installation from below.
pip install gdown

#Blurb Genre Collection
mkdir -p data/bgc
cd data/bgc
gdown "https://drive.google.com/uc?id=1AuB1qHWhqcD1LL3lV-2KmAdLJHnl14Dh"
tar -xzvf bgc.tar.gz
rm bgc.tar.gz
cd ../..

# Bibtex better stratified splits.
mkdir -p data/bibtex_stratified10folds_meka
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1V8OMs4HqxZCeZYoG2DbDK-8zGC-iL2wn' -O data/bibtex_stratified10folds_meka/Bibtex-fold1.arff
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1VCi72UEcpU0tGyN7xeaZbG0rxeiuTxP-' -O data/bibtex_stratified10folds_meka/Bibtex-fold2.arff
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1r0eFBbnEb15yCANBzBpqR7c7pJF53HI5' -O data/bibtex_stratified10folds_meka/Bibtex-fold3.arff
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=11LOW-yFq1EIM4sWy0XmMPPC7Rv2mjhFd' -O data/bibtex_stratified10folds_meka/Bibtex-fold4.arff
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1xo7IWMO6Cc0d6wkf051VvYm3W196Ns0P' -O data/bibtex_stratified10folds_meka/Bibtex-fold5.arff
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1meV9ldv9XRPL8Akb78G7TvsY_QcfmYqu' -O data/bibtex_stratified10folds_meka/Bibtex-fold6.arff
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=15stGx406I2jwySsfzHdUKvJhUxoDu6rH' -O data/bibtex_stratified10folds_meka/Bibtex-fold7.arff
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1AepSJKrQ_PA0yCUVsbU5gmlW6GRLDtCd' -O data/bibtex_stratified10folds_meka/Bibtex-fold8.arff
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1ATz5E7GkZ0Kh5i6_uBCqAp7bdMEGg7an' -O data/bibtex_stratified10folds_meka/Bibtex-fold9.arff
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1rvmycPjHD4Juvnf6DjwEBFhhNquheivy' -O data/bibtex_stratified10folds_meka/Bibtex-fold10.arff
