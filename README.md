# snaki
This repository is based on the code from https://github.com/vedantgoswami/SnakeGameAI used in https://www.geeksforgeeks.org/ai-driven-snake-game-using-deep-q-learning/

# How to run 
## Installation
    conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
    pip install pygame
    conda install matplotlib
    conda install IPython

## Instructions (copied from website)
To run this game first create an environment in the anaconda prompt or (any platform). Then install the necessary modules such as Pytorch(for DQ Learning Model), Pygame (for visuals of the game), and other basic modules.

Then run the agent.py file in the environment just created and then the training will start, and you will see the following two GUI one for the training progress and the other for the snake game driven by AI.

After achieving certain score you can quit the game and the model that you just trained will be stored in the path that you had defined in the save function of models.py.


