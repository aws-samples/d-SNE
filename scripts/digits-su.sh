#!/usr/bin/env bash
# Shell script for supervised domain adaptation experiments
# Experiments MNIST -> MNISTM
sh scripts/digits-mt-mm-su.sh
# Experiments MNIST <-> USPS
# sh scripts/digits-mt-us-su.sh
# Experiments MNIST <-> SVHN
sh scripts/digits-mt-sn-su.sh