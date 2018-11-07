#!/bin/bash

set -e

. python/setup/bootstrapVirtualEnv.sh

if [ -z ${1} ]; then
    # TODO: Improve usage
    echo "Please provide a python script as the first argument for this script ... "
    exit 1
fi

echo "========== Running python script ... =========="
echo ""
echo ""

python ${1}