# ppo.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from collections import deque
import random
from env import WebFuzzingEnvironment
from typing import List, Tuple, Dict


class PPONetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """
        PPO网络架构

        Args:
            input_dim (int): 输入维度
            hidden_dim (int): 隐藏层维度
            output_dim (int): 输出维度
        """
        super(PPONetwork, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            state (torch.Tensor): 状态输入

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (动作概率, 状态价值)
        """
        action_probs = self.actor(state)
        value = self.critic(state)
        return action_probs, value


class PPOAgent:
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 64,
                 lr: float = 0.001,
                 gamma: float = 0.99,
                 clip_epsilon: float = 0.2,
                 max_memory_size: int = 1000):
        """
        PPO智能体

        Args:
            state_dim (int): 状态空间维度
            action_dim (int): 动作空间维度
            hidden_dim (int): 隐藏层维度
            lr (float): 学习率
            gamma (float): 折扣因子
            clip_epsilon (float): PPO裁剪参数
            max_memory_size (int): 经验回放缓冲区大小
        """
        self.network = PPONetwork(state_dim, hidden_dim, action_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.memory = deque(maxlen=max_memory_size)
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma

        # 训练相关参数
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5

    def select_action(self, state: torch.Tensor) -> Tuple[int, float]:
        """
        选择动作

        Args:
            state (torch.Tensor): 当前状态

        Returns:
            Tuple[int, float]: (选择的动作, 动作概率)
        """
        self.network.eval()
        with torch.no_grad():
            action_probs, _ = self.network(state)
            dist = Categorical(action_probs)
            action = dist.sample()
            action_prob = action_probs[action.item()].item()
        self.network.train()
        return action.item(), action_prob

    def store_transition(self, transition: Tuple):
        """
        存储转换

        Args:
            transition (Tuple): (状态, 动作, 奖励, 下一状态, 动作概率)
        """
        self.memory.append(transition)

    def update(self, batch_size: int = 32, epochs: int = 10) -> Dict[str, float]:
        """
        更新策略网络

        Args:
            batch_size (int): 批次大小
            epochs (int): 每批数据的训练轮数

        Returns:
            Dict[str, float]: 训练相关的统计信息
        """
        if len(self.memory) < batch_size:
            return {
                'actor_loss': 0,
                'critic_loss': 0,
                'entropy': 0,
                'total_loss': 0
            }

        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([t[0] for t in batch])
        actions = torch.LongTensor([t[1] for t in batch])
        rewards = torch.FloatTensor([t[2] for t in batch])
        next_states = torch.FloatTensor([t[3] for t in batch])
        old_probs = torch.FloatTensor([t[4] for t in batch])

        # 计算优势函数
        with torch.no_grad():
            _, next_values = self.network(next_states)
            _, values = self.network(states)
            advantages = rewards + self.gamma * next_values.squeeze(-1) - values.squeeze(-1)
            returns = advantages + values.squeeze(-1)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO更新
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0

        for _ in range(epochs):
            # 计算新的动作概率和状态价值
            action_probs, values = self.network(states)
            dist = Categorical(action_probs)
            new_probs = dist.log_prob(actions).exp()
            entropy = dist.entropy().mean()

            # 计算比率
            ratio = new_probs / old_probs

            # PPO损失
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # 价值损失
            critic_loss = nn.MSELoss()(values.squeeze(-1), returns)

            # 总损失
            loss = (actor_loss
                    + self.value_loss_coef * critic_loss
                    - self.entropy_coef * entropy)

            # 优化器步骤
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.optimizer.step()

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy += entropy.item()

        # 计算平均损失
        avg_actor_loss = total_actor_loss / epochs
        avg_critic_loss = total_critic_loss / epochs
        avg_entropy = total_entropy / epochs

        return {
            'actor_loss': avg_actor_loss,
            'critic_loss': avg_critic_loss,
            'entropy': avg_entropy,
            'total_loss': avg_actor_loss + self.value_loss_coef * avg_critic_loss
        }


def train_fuzzer(
        target_url: str,
        episodes: int = 1000,
        max_steps: int = 100,
        batch_size: int = 32,
        update_frequency: int = 10,
        hidden_dim: int = 64,
        early_stopping_threshold: float = -0.1
) -> Dict[str, List[float]]:
    """
    训练Web Fuzzer

    Args:
        target_url (str): 目标URL
        episodes (int): 训练回合数
        max_steps (int): 每回合最大步数
        batch_size (int): 批次大小
        update_frequency (int): 策略更新频率
        hidden_dim (int): 隐藏层维度
        early_stopping_threshold (float): 提前停止阈值

    Returns:
        Dict[str, List[float]]: 训练历史数据
    """
    # 初始化环境和智能体
    env = WebFuzzingEnvironment(target_url)
    agent = PPOAgent(
        state_dim=env.observation_space_dim,
        action_dim=env.action_space_dim,
        hidden_dim=hidden_dim
    )

    # 训练历史记录
    history = {
        'episode_rewards': [],
        'coverage_counts': [],
        'actor_losses': [],
        'critic_losses': [],
        'entropies': []
    }

    # 移动平均奖励
    reward_window = deque(maxlen=100)

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            # 选择动作
            action, old_prob = agent.select_action(state)

            # 执行动作
            next_state, reward, done, info = env.step(action)

            # 存储转换
            transition = (state, action, reward, next_state, old_prob)
            agent.store_transition(transition)

            episode_reward += reward
            state = next_state

            # 定期更新策略
            if step % update_frequency == 0:
                update_info = agent.update(batch_size)
                history['actor_losses'].append(update_info['actor_loss'])
                history['critic_losses'].append(update_info['critic_loss'])
                history['entropies'].append(update_info['entropy'])

            if done:
                break

        # 记录回合统计信息
        history['episode_rewards'].append(episode_reward)
        history['coverage_counts'].append(len(env.coverage_history))
        reward_window.append(episode_reward)

        # 打印训练进度
        if (episode + 1) % 10 == 0:
            avg_reward = sum(reward_window) / len(reward_window)
            print(f"Episode {episode + 1}")
            print(f"Average Reward: {avg_reward:.2f}")
            print(f"Coverage Count: {len(env.coverage_history)}")
            print(f"Actor Loss: {history['actor_losses'][-1]:.4f}")
            print(f"Critic Loss: {history['critic_losses'][-1]:.4f}")
            print(f"Entropy: {history['entropies'][-1]:.4f}")
            print("-" * 50)

            # 提前停止检查
            if len(reward_window) == reward_window.maxlen and avg_reward < early_stopping_threshold:
                print("Early stopping due to poor performance")
                break

    return history


if __name__ == "__main__":
    # 训练参数
    TARGET_URL = "http://example.com/api"  # 替换为实际的目标URL
    TRAINING_CONFIG = {
        'episodes': 1000,
        'max_steps': 100,
        'batch_size': 32,
        'update_frequency': 10,
        'hidden_dim': 64,
        'early_stopping_threshold': -0.1
    }

    # 开始训练
    training_history = train_fuzzer(
        target_url=TARGET_URL,
        **TRAINING_CONFIG
    )

    # 可以添加训练结果的可视化代码
    print("Training completed!")
    print(f"Final coverage count: {training_history['coverage_counts'][-1]}")
    print(f"Average reward in last 100 episodes: {sum(training_history['episode_rewards'][-100:]) / 100:.2f}")
