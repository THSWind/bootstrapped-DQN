# Bootstrapped Deep Q-learning Network with Python and TensorFlow

WORK IN PROGRESS

This is a python implementation of bootstrapped DQN (that solves the detachment problem by using more intelligent exploration, or will do soon). Bootstrapped DQN essentially combines deep exploration with deep neural networks for exponentially faster learning. Deep exploration is exploring only when there are valuable learning opportunities, reasoned through the informational value of possible observation sequences with regard to long term consequences. At the start of each episode, bootstrapped DQN samples a single Q-value function from its approximate posterior. The agent then follows the policy which is optimal for that sample for the duration of the episode. 

Bootstrapped DQN has increased performance over regular DQN, and the detachment problem solved by [Go-Explore](https://arxiv.org/abs/1901.10995) refers to forgetting past frontiers which could lead to better rewards. Thus, solving this improves performance massively with the Atari-2600 games of Montezuma's Revenge and Pitfall! which are plagued by sparse rewards, so more intelligent exploration than used in other games is necessary here.

Network architecture is the same as used [here](https://arxiv.org/pdf/1509.06461.pdf).

### Repository structure ###

    .
    ├── README.md
    ├── requirements.txt
    ├── run.sh
    └── src
        ├── __init__.py
        ├── agent
        │   ├── __init__.py
        │   ├── explorers.py
        │   └── model.py
        ├── env
        │   └── montezuma_env.py
        ├── main.py
        └── utils
            ├── __init__.py
            ├── modules.py
            └── utils.py

### Usage ###



### To do ###
- Improve documentation
- Add ability to remember frontiers
- Debug and get this to work fully

