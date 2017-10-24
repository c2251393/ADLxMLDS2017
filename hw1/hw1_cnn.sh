#!/usr/bin/env bash
python3 predict.py $1 mfcc cnn cnn.mfcc.e500.h50.b100.l5.wx3.wy2.p2.pt -H 50 -b 100 -n 5 -o $2
