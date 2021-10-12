#!/bin/bash

# Activate the virtual environment
source /Users/yubyanj/opt/minority_game/bin/activate 

# Enumerate the experiments to run
EXPERIMENTS=( 1 2 3)

# Run Experiments
if ((${#EXPERIMENTS[@]}));then
    for experiment_number in ${EXPERIMENTS[@]};
    do
        python trainer.py --experiment $experiment_number &
    done
fi