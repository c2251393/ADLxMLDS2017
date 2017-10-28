#!/usr/bin/env bash
wget https://gitlab.com/c2251393/ADLhw1bestmodel/raw/master/brnn.all.e20.h1024.b4.l6.d0.5.pt -O models/brnn.all.e20.h1024.b4.l6.d0.5.pt
python3 predict.py $1 all brnn brnn.all.e20.h1024.b4.l6.d0.5.pt -H 1024 -b 4 -n 6 -d 0.5 -o $2
