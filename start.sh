#!/bin/bash

if [ -z "$1" ]
then 
    echo -ne "Insert value of n : "
    read n
else
    n=$1
fi

python3 main.py $n