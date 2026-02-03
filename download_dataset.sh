#!/bin/bash

set -e  # exit on error

curl -L -o csi-bench.zip \
  https://www.kaggle.com/api/v1/datasets/download/guozhenjennzhu/csi-bench

unzip csi-bench.zip -d data
rm csi-bench.zip
