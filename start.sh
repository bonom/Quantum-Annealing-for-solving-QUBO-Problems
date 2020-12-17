#!/bin/bash
clear
pip3 install -r requirements.txt | grep "not installed" 

#if [ -z "$1" ]
#then 
#    echo -ne "Insert value of n : "
#    read n
#else
#    n=$1
#fi

python3 -B start.py