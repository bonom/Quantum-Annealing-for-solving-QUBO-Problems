#!/bin/bash
clear
#pip3 install -r requirements.txt | grep "not installed" 

[ ! -d "output/" ] && mkdir output/

python3 -B start.py $1