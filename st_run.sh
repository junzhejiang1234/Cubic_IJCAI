#!/bin/bash

# Check if "market" argument is provided
if [ -z "$loc" ]; then
    echo "Usage: $0 --market <market_name>"
    exit 1
fi

# Change directory to the specified location
cd "./Execution/$market"

# Print the current working directory
echo "RUNNING EXP ON $(pwd) MARKET"

python main.py
