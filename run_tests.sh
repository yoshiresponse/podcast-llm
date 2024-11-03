#!/bin/bash
python -m pytest "$@" 

# Build documentation
echo "Building documentation..."
cd docs && make html
cd ..
