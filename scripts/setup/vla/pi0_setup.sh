#!/usr/bin/env bash

# OpenPI Setup Script
# Installs openpi into the current rio venv (third_party/openpi is cloned for local patching).
# Run from the repo root: bash scripts/setup/vla/pi0_setup.sh

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

echo "Starting OpenPI installation..."
echo "================================"

if [ ! -d "third_party/openpi" ]; then
    echo "Cloning openpi into third_party/openpi..."
    mkdir -p third_party
    GIT_LFS_SKIP_SMUDGE=1 git clone --recurse-submodules \
        https://github.com/physical-intelligence/openpi.git \
        third_party/openpi
else
    echo "third_party/openpi already exists, skipping clone."
fi

# Install openpi as an editable package into the rio venv (not openpi's own venv).
uv pip install -e third_party/openpi

# Install remaining pi0 deps.
uv pip install \
    "lerobot @ git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5" \
    "datasets==3.6.0" \
    "optax==0.2.6"

echo "================================"
echo "Pi0 installation complete."
