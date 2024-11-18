# Banana Navigation

This project solves the banana environment. It uses a neural network defined in [model.py](/model.py). The Agent class implements all the code needed to train the model given specific instantiation hyperparameters. The report details the specific architecture and how the model was trained in various ways.

Follow the instructions in the [Navigation](/Navigation.ipynb) notebook to install the dependencies. Due to the inability to find the correct environment online, the workspace was used to complete this project. For this reason, the environment path within the Agent class defaults to the recommended environment path.

To train an agent, simply instantiate it with the desired hyperparameters and call `agent.train_dqn(<checkpoint_name>)` where `<checkpoint_name>` is the file name for the saved final solution model learned during the training session. The session scores for each episode, along with the hyperperameters used for that training session are also stored in an SQLite database for later comparison. These are compared in the [Report](/Report.ipynb).

NOTE: if running outside the workpace, agent instantiation must use the correct environment path. This is the path where the banana Unity environment is located. As mentioned, the project is so outdated I could not find the correct Unity dependencies online. This means I could only run the project within the workspace with the older Unity dependencies and builds. On my system, the provided environment build was too old for the available Unity downloads. For this reason, only a basic README is needed, as it is expected that this project is ran on the provided Udacity workspace virtual machine with all dependencies pre-installed and the environment path already setup.
