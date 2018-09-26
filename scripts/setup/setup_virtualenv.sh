#!/bin/bash

set -e

################################################################################
################################## CONSTANTS  ##################################
################################################################################

WORKSPACE=./workspace
VIRTUALENV_DIR=${WORKSPACE}/venv
VIRTUALENV_NAME=automl

SETUP_DIR=scripts/setup/
SERVER_REQS_FILENAME=${SETUP_DIR}/server-requirements.txt

PYTHON_EXEC=python3.6

################################################################################
############################ CREATE THE ENVIRONMENT ############################
################################################################################
if [ -d ${VIRTUALENV_DIR}/${VIRTUALENV_NAME} ]; then
  echo "Environment ${VIRTUALENV_NAME} already exists. Skipping creation ..."
else
  virtualenv --python=${PYTHON_EXEC} ${VIRTUALENV_DIR}/${VIRTUALENV_NAME}
fi

################################################################################
###################### INSTALL ALL PACKAGES AS IN SERVER  ######################
################################################################################
echo "Enabling environment for current script ..."
source ${VIRTUALENV_DIR}/${VIRTUALENV_NAME}/bin/activate

PYTHON_USED=`which python`
PYTHON_DESIRED=${VIRTUALENV_DIR}/${VIRTUALENV_NAME}/bin/python

if [ "${PYTHON_USED}" != "${PYTHON_USED}" ]; then
  echo "Error while sourcing the environment ..."
  echo "Expected python is ${PYTHON_DESIRED} but got ${PYTHON_USED}. Exit."
  exit 1
fi

pip install -r ${SERVER_REQS_FILENAME}
