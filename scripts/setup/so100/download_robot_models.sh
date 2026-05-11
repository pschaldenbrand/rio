#!/bin/bash

#check if script is run from correct location
if [ ! -d "scripts/setup/so100" ] ; then
    echo -e "\033[31mPlease run this script from the root directory of the repository.\033[0m"
    exit 1
fi
# Get SO100 URDF models
mkdir tmp
mkdir -p assets/robots/
cd tmp
wget https://github.com/TheRobotStudio/SO-ARM100/archive/refs/heads/main.zip
unzip main.zip "SO-ARM100-main/Simulation/SO100/*"
unzip main.zip "SO-ARM100-main/Simulation/SO101/*"

# check if assets/robots/so100 exists
if [ -d "../assets/robots/SO100" ] ; then
    echo -e "\033[33mDirectory ../assets/robots/SO100 already exists. Skipping download.\033[0m"
else
    mv SO-ARM100-main/Simulation/SO100 ../assets/robots/
fi

if [ -d "../assets/robots/SO101" ] ; then
    echo -e "\033[33mDirectory ../assets/robots/SO101 already exists. Skipping download.\033[0m"
else
    mv SO-ARM100-main/Simulation/SO101 ../assets/robots/
fi
cd ..
rm -rf tmp