#!/usr/bin/env bash
bash download_pretrained.sh
python3 generate.py -test $1 -n 100 models/D1.G2.n100.e1800.pt
