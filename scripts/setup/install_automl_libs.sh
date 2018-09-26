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
AUTOML_REQS_FILENAME=${SETUP_DIR}/automl-requirements.txt

PYTHON_EXEC=python3.6


################################################################################
######################## ENABLE THE VIRTUAL ENVIRONMENT ########################
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

################################################################################
########################### INSTALL ALL DEPENDENCIES ###########################
################################################################################

# Install RRF manually on mac. As mentioned in auto-sklearn's instructions
RRF_HASH_URL=c555cfa3c7d0078dded091d4901ed52344bbb925077aa70b871faf35fd58
RRF_VERSION=0.7.4
RRF_TARGZ_NAME=pyrfr-${RRF_VERSION}.tar.gz

RRF_URL=https://files.pythonhosted.org/packages/c3/c6/${RRF_HASH_URL}/${RRF_TARGZ_NAME}

if [ ! -f ${WORKSPACE}/${RRF_TARGZ_NAME} ]; then
  curl -L ${RRF_URL} -o ${WORKSPACE}/${RRF_TARGZ_NAME}
fi

cd ${WORKSPACE}
tar xzf ${RRF_TARGZ_NAME}
EXTRACT_DIR=${RRF_TARGZ_NAME%.*}
EXTRACT_DIR=${EXTRACT_DIR%.*}

cd ${EXTRACT_DIR}
cp ${INIT_DIR}/scripts/setup/pyrfr_setup.py setup.py
python setup.py install

# Install XGBoost
pip install xgboost # Install it manually

# Auto-sklearn. 
# As instructed in https://automl.github.io/auto-sklearn/stable/installation.html
AUTOSKLEARN_REQS_FILE=https://raw.githubusercontent.com/automl/auto-sklearn/master/requirements.txt
curl ${AUTOSKLEARN_REQS_FILE} | xargs -n 1 -L 1 pip install
pip install -r ${AUTOML_REQS_FILENAME}

# TPOT. 
# As instructed in http://epistasislab.github.io/tpot/installing/
TPOT_REQS_FILE=${INIT_DIR}/scripts/setup/tpot-requirements.txt
pip install -r ${TPOT_REQS_FILE}

################################################################################
######################### INSTALL THE AUTOML PACKAGES  #########################
################################################################################
AUTOML_REQS_FILE=${INIT_DIR}/scripts/setup/automl-requirements.txt
pip install -r ${AUTOML_REQS_FILE}