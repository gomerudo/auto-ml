#!/bin/bash

set -e

################################################################################
################################## CONSTANTS  ##################################
################################################################################

PYTHON_VERSION=3.4.6
PYTHON_TGZ_NAME=Python-${PYTHON_VERSION}.tgz

PYTHON_SERVER=https://www.python.org
PYTHON_DOWNLOAD_DIR=/tmp

DOWNLOAD_URL=${PYTHON_SERVER}/ftp/python/${PYTHON_VERSION}/${PYTHON_TGZ_NAME}

################################################################################
###################### DOWNLOAD THE BINARIES IF NOT THERE ######################
################################################################################

if [ -f ${PYTHON_DOWNLOAD_DIR}/${PYTHON_TGZ_NAME} ]; then
  echo "Binaries for Python ${PYTHON_VERSION} have already been downloaded ..."
else
  echo "Downloading binaries as ${PYTHON_DOWNLOAD_DIR}/${PYTHON_TGZ_NAME} ... "
  curl -L ${DOWNLOAD_URL} -o ${PYTHON_DOWNLOAD_DIR}/${PYTHON_TGZ_NAME}
fi

################################################################################
################ INSTALL PYTHON AS AN ALTERNATIVE INSTALLATION  ################
################################################################################

cd ${PYTHON_DOWNLOAD_DIR}
tar xzf ${PYTHON_TGZ_NAME}
cd ${PYTHON_TGZ_NAME%.*}

echo "Working in $(pwd) ..."

# Exporting some flags for SSL/TLS. Otherwise, next error will be thrown :
#  - Ignoring ensurepip failure: pip 9.0.1 requires SSL/TLS

# For OS X only
export CPPFLAGS=-I$(brew --prefix openssl)/include
export LDFLAGS=-L$(brew --prefix openssl)/lib

./configure --enable-optimizations
make altinstall
