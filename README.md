# Lunar Lander using Deep Q Learning in OpenAI GYM

This project leverages **Deep Q-Networks (DQN)**, a reinforcement learning algorithm, to train an autonomous agent in OpenAI's Lunar Lander environment. The goal is to achieve stable and efficient landings through Q-learning combined with deep neural networks.

### HOW TO RUN THE CODE
First, cd to the direct folder. And then run this code in terminal.
```bash
 
python3.10 main_lunar_lander.py
 
```

## DEMO
![](https://github.com/EricChen0104/DQN_Lunar_Lander_gym/blob/master/plt/best_play306.gif)

## What is [OpenAI GYM](https://gymnasium.farama.org/)
OpenAI gym is a research environment for reinforcement learning(RL) which was created by OpenAI. It prevent a simulation environment to make the researcher design, train or even test any of the RL algorithm more convenient. 

Gym supports many kinds of classical control problem. For instance, Rotary Inverted Pendulum, Discrete Action Space(like CartPole, MountainCar). Gym even includes high level vision and continuous control mission(like Atari and MuJoCo).

The main idea of Gym is to emphasize the simplicity and scalability. It's API is extremely intuitive, it contains four main steps: initialize environment, reset, interaction(step) and render. Makes user focus on algorithm's develop and improvment since the developers do not need to spent time on handling the environment detail.

## WHAT IS DEEP Q LEARNING
Before we learn Qeep Q Learning, we must know Q Learning algorithm.
### Q Learning
Q Learning is a classic algorithm in reinforcement learning, let the Agent continue trying to get the best strategy for the next action. 

The core concepts of this algorithm:

1. Q value
  - means the long term reward for the 'chosen action' in a 'certain state'
  - formula: <br> ![截圖 2025-06-03 晚上11 57 44](https://github.com/user-attachments/assets/c42121c2-d1b7-48b3-9a75-9b1fb9ad7f0d)
    * α (learning rate): control the update range
    * γ (loss factor): the importance of the future reward
2. update rule
  - Agent will keep update the Q-Table through interactive with the environment. Eventually, the Agent will learn 'which action in which statement can get the largest reward'.
3. Model-Free
  - Do not need to know how the environment works, just simply learn from the environment reward.

Work flow:
1. Agent chose a action in a statement(ex: ε-greedy strategy)
2. After the implement action, the agent gets a reward and enter new statement
3. update the Q value following the **Bellman Equation**:
  - new Q value = old Q value + learning rate * (realtime reward + loss of the future maxium reward - old Q value)

### Deep Q Learning
So, after we know how the Q Learning works. The Deep Q Learning seems very simple.

Deep Q Learning is a updated version of Q Learning. By using Deep Neutral Network the replace the Q-Table to solve the problem of "too many state" (ex: game screen, high demention data).


  
## WHY DO THIS PROJECT
