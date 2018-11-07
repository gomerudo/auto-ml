#!/bin/bash

set -e

################################################################################
################################# GLOBAL VARS  #################################
################################################################################

# Directories
INIT_DIR=`pwd`
WORKSPACE=${INIT_DIR}/pyworkspace

# Virtualenv constants
VIRTUALENV_VERSION=16.0.0
VIRTUALENV_DIRNAME=virtualenv-${VIRTUALENV_VERSION}

# Custom virtual environment name
MY_VENV_NAME=automl

################################################################################
##################################### MAIN #####################################
################################################################################

# Start always in the INIT_DIR - the place where the script was called 
. ${WORKSPACE}/${VIRTUALENV_DIRNAME}/${MY_VENV_NAME}/bin/activate

pip install requests
