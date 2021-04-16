#!/bin/bash
clear

[ ! -d "output/" ] && mkdir output/

python3 -B start.py $1