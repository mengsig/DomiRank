#!/bin/bash

FILENAME_ENV="$(pwd)/domirank_venv"

if [ ! -d "$FILENAME_ENV" ]; then
    echo "Could not locate the virtual environment '$FILENAME_ENV'. Please re-install using:
    rm -r coarsening
    bash install.sh

If the error persists, re-pull the repository using:
    git pull git@github.com:mengsig/DomiRank.git"
    exit 0
fi

echo "Activating virtual environment '$FILENAME_ENV'."
source $FILENAME_ENV/bin/activate


