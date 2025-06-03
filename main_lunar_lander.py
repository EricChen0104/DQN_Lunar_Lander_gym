import gym
from DQN import Agent
from utils import plot_learning_curve
import numpy as np
import os 
import imageio

def save_gif(frames, filename, fps=30):
    """
    將畫面序列儲存為 GIF
    :param frames: 畫面列表（RGB 陣列）
    :param filename: GIF 檔案路徑
    :param fps: 每秒幀數
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with imageio.get_writer(filename, mode='I', fps=fps, format='GIF') as writer:
        for frame in frames:
            writer.append_data(frame)

def test_model(n_games):
    env = gym.make('LunarLander-v2', render_mode='rgb_array')

    save_dir = './models'
    model_path = os.path.join(save_dir, f'lunar_lander_dqn_305_-58.32.pth')
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4,
            eps_end=0.01, input_dims=[8], lr=0.003)
    agent.load_model(model_path)

    scores = []

    best_score = -100000

    for i in range(n_games):
        score = 0
        done = False
        observation, _ = env.reset()
        frames = []
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            score += reward

            frame = env.render()
            frames.append(frame)
            observation = observation_
        scores.append(score)
        best_score = max(best_score, score)
        print('episode: ', i+1, 'score: ', score, 'best score: ', best_score)
        gif_filename = f'./plt/test/play{i+1}.gif'
        save_gif(frames, gif_filename)

def train_model():
    env = gym.make('LunarLander-v2', render_mode='rgb_array')
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4,
            eps_end=0.01, input_dims=[8], lr=0.003)
        
    scores, eps_history = [], []

    n_games = 500

    save_dir = './models'  # 模型儲存的目錄
    save_dir_plt = './plt'  # 模型儲存的目錄
    os.makedirs(save_dir, exist_ok=True)  # 確保儲存目錄存在

    best_score = -100000

    

    for i in range(n_games):
        score = 0
        done = False
        observation, _ = env.reset()
        frames = []
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            score += reward

            frame = env.render()
            # print(frame)
            frames.append(frame)

            agent.store_transition(observation, action, reward, observation_, done)

            agent.learn()
            observation = observation_
        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])
        print('episode: ', i, 'score %.2f ' % score, 'best score %.2f' % best_score, 'avgerage score %.2f' % avg_score, 'epsilon %.2f ' % agent.epsilon)

        # 儲存模型的條件：當平均分數提升時儲存
        if score > best_score and i >= 50:  # 確保有足夠回合計算平均分數
            best_score = score
            model_path = os.path.join(save_dir, f'lunar_lander_dqn_{i}_{avg_score:.2f}.pth')
            agent.save_model(model_path)  # 呼叫 Agent 的儲存方法
            model_path = os.path.join(save_dir, f'best_model.pth')
            agent.save_model(model_path)  # 呼叫 Agent 的儲存方法
            gif_filename = f'./plt/best_play{i+1}.gif'
            save_gif(frames, gif_filename)
            print(f'Saved model at episode {i} with avg score {avg_score:.2f}')

    x = [i+1 for i in range(n_games)]
    filename = os.path.join(save_dir_plt, 'lunar_lander.png')  # 儲存到 models 目錄
    plot_learning_curve(x, scores, eps_history, filename)


if __name__ == '__main__':
    n_games = 5
    test_model(n_games)