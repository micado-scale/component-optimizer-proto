#!/bin/bash

set -ex

PDIR=env/optimiser

echo "Reseting '$PDIR'"

rm -rf "$PDIR"

virtualenv -p python3.6 "$PDIR"
source "$PDIR"/bin/activate
pip install --upgrade virtualenv
pip install --upgrade pip
pip install -r requirements.txt 

set +ex

echo "Now, the virtualenv is ready. To activate it, run this:"
echo "source '$PDIR/bin/activate'"


