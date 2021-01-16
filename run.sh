clear
rm -rf build

mkdir build
cd build

cmake ../
make

FILE=./several_days_of_cuda
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

if [ -f "$FILE" ]; then
    ./several_days_of_cuda
    cd ../
    rm -rf build
    printf "\n\n\n${GREEN}Compilation Successful!!!\n"
else
    printf "\n\n\n${RED}Compilation Failure!!!\n"
fi
