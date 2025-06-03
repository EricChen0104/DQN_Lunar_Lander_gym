import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(episodes, scores, eps_history, filename):
    """繪製並儲存學習曲線（分數和 epsilon）"""
    fig, ax1 = plt.subplots()

    # 繪製分數曲線
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score', color='tab:blue')
    ax1.plot(episodes, scores, color='tab:blue', label='Score')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # 創建第二個 y 軸繪製 epsilon
    ax2 = ax1.twinx()
    ax2.set_ylabel('Epsilon', color='tab:red')
    ax2.plot(episodes, eps_history, color='tab:red', label='Epsilon')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # 添加圖例
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)

    # 設置標題和佈局
    plt.title('Learning Curve: Score and Epsilon vs Episode')
    fig.tight_layout()

    # 儲存圖表
    plt.savefig(filename, bbox_inches='tight')
    print(f'Learning curve saved to {filename}')
    
    # 顯示圖表
    plt.show()
