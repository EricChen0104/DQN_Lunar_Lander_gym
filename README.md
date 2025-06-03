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

the core concepts of this algorithm:

1. the bottleneck of Q Learning:
   - tradition Q-Learning save the Q(s,a) by the Q-Table, however, the table will explode when there are too many states.
   - for example: Atari game's state is 210 x 160 pixels, it can not even use a table to save.
2. the specialize of DQN
   - it use neutral network to predict *Q value*: input a statement, then output all of the actions.
3. the key technique:
   - Experience Replay: to avoid the relation of the continuous data by saving the past experience and stochastic sampling training.
   - Target Network: use another network to calculate the target Q value. 

  
## WHY DO THIS PROJECT
For those new to reinforcement learning, Deep Q-Learning (DQN) is an excellent project to learn and practice. It combines foundational RL concepts (like Q-values and exploration-exploitation) with deep learning, while being approachable enough for beginners.

Using OpenAI Gym simplifies the process further:

🚀 Pre-built environments (like LunarLander-v2) handle physics simulation and rendering.

🏆 Built-in reward functions eliminate the need to design rewards from scratch (often the hardest part of RL projects).

📊 Standardized benchmarks make it easy to compare your agent’s performance.

Why DQN Fits the Lunar Lander Environment
The Lunar Lander task (landing a spacecraft safely on a target pad) is a perfect match for DQN because:

1. Handles High-Dimensional State Space
   - The environment provides observations like position, velocity, and terrain angles (8-dimensional state).
   - DQN’s neural network can process these continuous states without needing manual feature engineering.

2. Discrete Action Space
   - Lunar Lander has 4 discrete actions (fire engines left/right/down or do nothing), ideal for Q-learning’s action-selection approach.

3. Balanced Complexity
   - Challenging enough to require deep learning (simple Q-tables fail), but small enough to train on a laptop.
   - Faster to experiment with than pixel-based games (e.g., Atari).

4. Interpretable Training
   - Rewards are dense (every step gives feedback), helping the agent learn faster.
   - Visualizing the lander’s behavior is intuitive (vs. abstract games).

### Key Implementation Notes
While OpenAI Gym provides the reward function, you’ll still need to:
- Tune hyperparameters (e.g., learning rate, discount factor γ).
- Implement experience replay and target networks (critical for stability).
- Manage exploration (e.g., ε-greedy decay) to balance random actions vs. learned policy.

### Next Steps
Try improving the vanilla DQN with:
- Double DQN (reduces overestimation of Q-values).
- Prioritized Experience Replay (replays important transitions more often).

The code is for the beginners as a example of how to implement and train the model effiecently using Deep Q Learning. Hope you guys love it!!
