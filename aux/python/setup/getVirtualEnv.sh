#!/bin/bash

set -e

################################################################################
################################# GLOBAL VARS  #################################
################################################################################

# Directories
INIT_DIR=`pwd`
DEST_DIR=${INIT_DIR}/pyworkspace
TMP_DIR=/tmp

# Virtualenv constants
VIRTUALENV_VERSION=16.0.0
VIRTUALENV_PKG_NAME=virtualenv
VIRTUALENV_EXT=.tar.gz

# Custom virtual environment name
MY_VENV_NAME=automl

# PyPi constants
PYPY_BASE_URL=https://pypi.python.org/packages/source/v
VIRTUALENV_PYPI_PKG_NAME=${VIRTUALENV_PKG_NAME}-${VIRTUALENV_VERSION}${VIRTUALENV_EXT}
VIRTUALENV_PYPI_URL=${PYPY_BASE_URL}/virtualenv/${VIRTUALENV_PYPI_PKG_NAME}

################################################################################
##################################### MAIN #####################################
################################################################################

# Start always in the INIT_DIR - the place where the script was called 
cd ${INIT_DIR}
mkdir -p ${DEST_DIR}

# Download the file in tmp directory 
if [ ! -d  ${TMP_DIR}/${VIRTUALENV_PYPI_PKG_NAME} ]; then
    echo "Downloading ${VIRTUALENV_PYPI_URL} into ${TMP_DIR}/${VIRTUALENV_PYPI_PKG_NAME}"
    curl -L ${VIRTUALENV_PYPI_URL} -o ${TMP_DIR}/${VIRTUALENV_PYPI_PKG_NAME}
fi


# Untar in the same directory 
if [ ! -d ${DEST_DIR}/${VIRTUALENV_PKG_NAME}-${VIRTUALENV_VERSION} ];  then
    cd ${DEST_DIR}
    tar xvfz ${TMP_DIR}/${VIRTUALENV_PYPI_PKG_NAME}
fi



# Browse to the directory
cd ${DEST_DIR}/${VIRTUALENV_PKG_NAME}-${VIRTUALENV_VERSION}

# Install it with python3 all the time

echo "Creating the virtual environment"
if [ ! -d ${MY_VENV_NAME} ]; then
    python3 virtualenv.py ${MY_VENV_NAME}
else
    echo "Virtual environment ${MY_VENV_NAME} already exists. Skipping creation ..."
fi

cd ${INIT_DIR}

exit 0