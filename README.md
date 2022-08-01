# Introduction
"Snaki" is a project in which we try to compare the reinforcement learning methods DQN, SARSA and PPO in a video game snake environment. A detailed documentation can be found in the file 'Dokumentation.docx'. A list of run experiments can be found in the Excel file 'Experimentenkatalog.xlsx'.

# Installation
1. Create python environment. These steps are done with the regular python virtual environment. (Alternatively you can use Anaconda) 

    python -m snaki-env

1. activate python environment

    source ./snaki-env/bin/activate

1. (Optional, but recommended) use CUDA toolkit for faster training with a gpu (see https://pytorch.org/get-started/previous-versions/ for possible installations). Change the version in the command below to your installed CUDA version. Example command for CUDA 11.3:

    pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

1. Install pip requirements

    pip install pygame matplotlib IPython

1. Install ppo requirements

    pip install stable-baselines3
    pip install -e ./Snake


# How to run
Start the virtual environment. If not using Anaconda, run this command:

    source ./snaki-env/bin/activate

Before training, make sure to adjust the parameters in the file 'parameters.py'. Some important parameters:
* 'method': select which agent to run.
* 'train': determines if a training or test run is done. A training run overwrites the currently saved models of the neural networks. The agent 'euclidean' can not be trained.
* 'show_gui': wheter or not the GUI for the game is displayed. Disabling this greatly improves training speed.
* 'device': whether to use the CPU or CUDA-enabled GPU for training.

Other parameters are hyperparameters for the agents or game settings.

Start a training or test run by running the python script 'start_training.py". This script automatically loads all the parameters from the python script 'parameters.py'.
The trained models are saved in the folder 'saved_models_X' based on the select agent. Logs can be found in the 'logs' folder.

(optional): the game can be manually played with the "snake_game_manual.py" script.

