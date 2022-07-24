########################## CURRENT ONE
# Basics
This repository is based on the code from https://github.com/vedantgoswami/SnakeGameAI used in https://www.geeksforgeeks.org/ai-driven-snake-game-using-deep-q-learning/.


# Installation with Anaconda
1. Create new Anaconda environment with python 3.9

1. (Optional) use CUDA toolkit for faster training with a gpu (see https://pytorch.org/get-started/previous-versions/ for possible installations). Change the version in the command below to your installed CUDA version. Example command for CUDA 11.3:

    conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

1. Install pip in the environment 

    conda install pip

1. Install pip requirements

    pip install pygame matplotlib IPython

# How to run
Run the agent.py file in the environment just created and then the training will start, and you will see the following two GUI one for the training progress and the other for the snake game driven by AI.

After achieving certain score you can quit the game and the model that you just trained will be stored in the path that you had defined in the save function of models.py.



########################## OLD ONE
# How to run 
# Installation OLD
    conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
    conda install pip
    pip install pygame matplotlib IPython
    conda install matplotlib
    conda install IPython

## Instructions (copied from website)
To run this game first create an environment in the anaconda prompt or (any platform). Then install the necessary modules such as Pytorch (for DQ Learning Model), Pygame (for visuals of the game), and other basic modules.

Then run the agent.py file in the environment just created and then the training will start, and you will see the following two GUI one for the training progress and the other for the snake game driven by AI.

After achieving certain score you can quit the game and the model that you just trained will be stored in the path that you had defined in the save function of models.py.


