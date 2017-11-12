#!/usr/bin/env bash
time python3 predict.py $1 s2vt.h512.b16.e60.pt $2 nan -H 512 -d 0.3 -B 2 -x
