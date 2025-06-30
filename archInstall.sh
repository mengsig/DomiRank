#!bin/bash

if [ ! -x "$0" ]; then
    chmod +x "$0"
    bash "$0"
    exit 0
fi

echo "Installing necessary python and python-virtualenv..."
#sudo pacman -S --needed base-devel
#yay -S python310
echo "Installed python3 and python-virtualenv."

if ! command -v python3.10 &>/dev/null; then
    echo "Python3 installation failed. Exiting..."
    exit 1
fi

VENV_NAME="domirank_venv"

if [ -d "$VENV_NAME" ]; then
    echo "Virtual environment already exists. Skipping creation..."
else
    echo "Creating virtual environment named: '$VENV_NAME'..."
    python3.10 -m venv "$VENV_NAME"
    echo "Virtual environment '$VENV_NAME' created."
fi

if [ -n "$VIRTUAL_ENV" ] && [ "$VIRTUAL_ENV" == "$(pwd)/$VENV_NAME" ]; then
    echo "Already inside the virtual environment '$VENV_NAME'"
else
    echo "Activating virtual environment: '$VENV_NAME'..."
    source "$VENV_NAME/bin/activate"
fi

echo "Upgrading pip..."
pip install --upgrade pip setuptools
echo "Upgraded pip."

if [ -f "requirements.txt" ]; then
    echo "Installing python dependencies..."
    pip install -r requirements.txt
    echo "Finished installing dependencies."
else
    echo "Warning: requirements.txt not found. Exiting installation..."
    echo "Please consider a re-pull of the repository and try installing again. Simply run:
    git pull git@github.com:mengsig/DomiRank"
    exit 1
fi

pip install -e .

echo "Setup Complete! Virtual environment '$VENV_NAME' has now been successfully created."

FILENAME_ENV="$(pwd)/env.sh"
if [ ! -f "$FILENAME_ENV" ]; then
    echo "Cannot find the '$FILENAME_ENV' file. Please re-pull via:
    git pull git@github.com:mengsig/DomiRank"
    echo "Stopping installation..."
    exit 1
fi

if [ ! -x "$FILENAME_ENV" ]; then
    chmod +x "$FILENAME_ENV"
    echo "Created executable name '$FILENAME_ENV'"
fi

echo "To activate the virtual environment at any time, run the command:
    source env.sh"



