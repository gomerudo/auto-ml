#!/bin/bash

set -e

################################################################################
################################## CONSTANTS  ##################################
################################################################################

INIT_DIR=`pwd`
WORKSPACE=${INIT_DIR}/workspace
VIRTUALENV_DIR=${WORKSPACE}/venv
VIRTUALENV_NAME=automl

SETUP_DIR=scripts/setup/

################################################################################
######################## ENABLE THE VIRTUAL ENVIRONMENT ########################
################################################################################
echo "Enabling environment for current script ..."
source ${VIRTUALENV_DIR}/${VIRTUALENV_NAME}/bin/activate

pip install ipykernel
ipython kernel install --user --name=${VIRTUALENV_NAME}