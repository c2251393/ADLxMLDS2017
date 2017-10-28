#!/usr/bin/env bash
python predict.py $1 mfcc brnn brnn.mfcc.e400.h100.b100.l6.d0.5.pt -H 100 -b 100 -n 6 -d 0.5 -o $2
