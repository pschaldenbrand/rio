#!/bin/bash

DATA_DIR=${1:-"data/demos"}
for demo_file in ${DATA_DIR}/*.vla; do
    echo "Playing demo: ${demo_file}"
    uv run -m examples.replay_data --loader-cfg.path "${demo_file}"
done