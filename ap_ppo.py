# ap_ppo.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from collections import deque, defaultdict
import random
import asyncio
import time
from typing import List, Tuple, Dict, Optional
from env import PublicTestEnvironment, WebFuzzingEnvironment


class PrioritizedReplayBuffer:
    """优先级经验回放缓冲区"""

    def __init__(self, max_size: int, alpha: float = 0.6, beta: float = 0.4):
        self.max_size = max_size
        self.alpha = alpha  # 优先级指数
        self.beta = beta  # 重要性采样指数
        self.buffer = deque(maxlen=max_size)
        self.priorities = deque(maxlen=max_size)
        self.episode_boundaries = defaultdict(list)  # 记录每个episode的边界

    def add(self, transition: Tuple, priority: float, episode_id: int):
        """添加新的转换到缓冲区"""
        self.buffer.append(transition)
        self.priorities.append(priority)
        self.episode_boundaries[episode_id].append(len(self.buffer) - 1)

    def sample(self, batch_size: int, current_episode: int) -> Tuple[List, np.ndarray]:
        """优先级采样"""
        priorities = np.array(self.priorities) ** self.alpha

        # 计算每个episode的平均优先级
        episode_priorities = defaultdict(float)
        for ep_id, indices in self.episode_boundaries.items():
            if indices:
                episode_priorities[ep_id] = np.mean([priorities[i] for i in indices])

        # 根据episode优先级调整当前episode的采样概率
        current_ep_bonus = 1.5 if current_episode in episode_priorities else 1.0
        if current_episode in episode_priorities:
            episode_priorities[current_episode] *= current_ep_bonus

        # 计算采样概率
        probs = priorities / priorities.sum()

        # 使用episode优先级调整采样概率
        for ep_id, indices in self.episode_boundaries.items():
            ep_priority = episode_priorities[ep_id]
            for idx in indices:
                probs[idx] *= ep_priority

        probs /= probs.sum()  # 重新归一化

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        batch = [self.buffer[idx] for idx in indices]
        return batch, weights

    def update_priorities(self, indices: List[int], priorities: List[float]):
        """更新优先级"""
        for idx, priority in zip(indices, priorities):
            if idx < len(self.priorities):
                self.priorities[idx] = priority + 1e-6


class MultiObjectiveNetwork(nn.Module):
    """多目标网络架构"""

    def __init__(self, input_dim: int, hidden_dim: int, action_dim: int):
        super(MultiObjectiveNetwork, self).__init__()

        # 共享特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # 动作策略头
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

        # 价值评估头
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # 覆盖率预测头
        self.coverage_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(state)
        action_probs = self.actor(features)
        value = self.critic(features)
        coverage_pred = self.coverage_predictor(features)
        return action_probs, value, coverage_pred


class AdaptivePriorityPPO:
    """自适应优先级PPO算法"""

    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            hidden_dim: int = 64,
            lr: float = 0.001,
            gamma: float = 0.99,
            clip_epsilon: float = 0.2,
            max_memory_size: int = 10000,
            coverage_weight: float = 0.5,
            entropy_weight: float = 0.01
    ):
        self.network = MultiObjectiveNetwork(state_dim, hidden_dim, action_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.memory = PrioritizedReplayBuffer(max_memory_size)

        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        self.coverage_weight = coverage_weight
        self.entropy_weight = entropy_weight

        # 自适应参数
        self.temp = 1.0  # 温度参数
        self.coverage_threshold = 0.5  # 覆盖率阈值
        self.success_rate = deque(maxlen=100)  # 成功率窗口

    def select_action(self, state: torch.Tensor, training: bool = True) -> Tuple[int, float, float]:
        """选择动作"""
        self.network.eval()
        with torch.no_grad():
            action_probs, _, coverage_pred = self.network(state)

            if training:
                # 根据覆盖率预测调整探索
                if coverage_pred.item() > self.coverage_threshold:
                    self.temp = max(0.5, self.temp * 0.995)  # 减少探索
                else:
                    self.temp = min(2.0, self.temp * 1.005)  # 增加探索

                # 使用温度参数调整动作概率
                adjusted_probs = action_probs ** (1 / self.temp)
                adjusted_probs = adjusted_probs / adjusted_probs.sum()

                dist = Categorical(adjusted_probs)
                action = dist.sample()
                action_prob = adjusted_probs[action.item()].item()
            else:
                action = torch.argmax(action_probs)
                action_prob = 1.0

        self.network.train()
        return action.item(), action_prob, coverage_pred.item()

    def calculate_priority(self, reward: float, coverage_gain: float) -> float:
        """计算转换优先级"""
        return abs(reward) + self.coverage_weight * coverage_gain

    def update(self, batch_size: int = 32, epochs: int = 10,
               current_episode: int = 0) -> Dict[str, float]:
        """更新策略"""
        if len(self.memory.buffer) < batch_size:
            return {
                'actor_loss': 0,
                'critic_loss': 0,
                'coverage_loss': 0,
                'entropy': 0,
                'total_loss': 0
            }

        total_actor_loss = 0
        total_critic_loss = 0
        total_coverage_loss = 0
        total_entropy = 0

        for _ in range(epochs):
            # 优先级采样
            batch, weights = self.memory.sample(batch_size, current_episode)
            weights = torch.FloatTensor(weights)

            states = torch.FloatTensor([t[0] for t in batch])
            actions = torch.LongTensor([t[1] for t in batch])
            rewards = torch.FloatTensor([t[2] for t in batch])
            next_states = torch.FloatTensor([t[3] for t in batch])
            old_probs = torch.FloatTensor([t[4] for t in batch])
            coverage_gains = torch.FloatTensor([t[5] for t in batch])

            # 前向传播
            action_probs, values, coverage_preds = self.network(states)
            _, next_values, _ = self.network(next_states)

            # 计算优势和回报
            advantages = rewards + self.gamma * next_values.squeeze(-1) - values.squeeze(-1)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            returns = advantages + values.squeeze(-1)

            # PPO损失计算
            dist = Categorical(action_probs)
            new_probs = dist.log_prob(actions).exp()
            entropy = dist.entropy().mean()

            ratio = new_probs / old_probs
            surr1 = ratio * advantages * weights
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages * weights
            actor_loss = -torch.min(surr1, surr2).mean()

            # 价值损失
            critic_loss = nn.MSELoss()(values.squeeze(-1), returns) * weights.mean()

            # 覆盖率预测损失
            coverage_loss = nn.BCELoss()(coverage_preds.squeeze(-1), coverage_gains)

            # 总损失
            loss = (actor_loss +
                    0.5 * critic_loss +
                    self.coverage_weight * coverage_loss -
                    self.entropy_weight * entropy)

            # 优化器步骤
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()

            # 更新优先级
            td_errors = (advantages * weights).abs().detach().numpy()
            self.memory.update_priorities(range(len(batch)), td_errors)

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_coverage_loss += coverage_loss.item()
            total_entropy += entropy.item()

        return {
            'actor_loss': total_actor_loss / epochs,
            'critic_loss': total_critic_loss / epochs,
            'coverage_loss': total_coverage_loss / epochs,
            'entropy': total_entropy / epochs,
            'total_loss': (total_actor_loss + 0.5 * total_critic_loss +
                           self.coverage_weight * total_coverage_loss) / epochs
        }


async def train_ap_ppo(
        test_env: PublicTestEnvironment,
        episodes: int = 50,
        max_steps: int = 25,
        batch_size: int = 16,
        hidden_dim: int = 64,
        early_stopping_patience: int = 20
) -> Dict[str, List[float]]:
    """训练AP-PPO"""

    env = WebFuzzingEnvironment(test_env)
    agent = AdaptivePriorityPPO(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        hidden_dim=hidden_dim
    )

    history = {
        'episode_rewards': [],
        'coverage_counts': [],
        'actor_losses': [],
        'critic_losses': [],
        'coverage_losses': [],
        'entropies': []
    }

    best_reward = float('-inf')
    no_improvement_count = 0

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        prev_coverage = len(test_env.coverage_history)

        for step in range(max_steps):
            # 选择动作
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action, action_prob, coverage_pred = agent.select_action(state_tensor)

            # 执行动作
            next_state, reward, done, info = await env.step(action)

            # 计算覆盖率增益
            current_coverage = len(test_env.coverage_history)
            coverage_gain = float(current_coverage > prev_coverage)

            # 计算优先级并存储转换
            priority = agent.calculate_priority(reward, coverage_gain)
            transition = (state, action, reward, next_state, action_prob, coverage_gain)
            agent.memory.add(transition, priority, episode)

            episode_reward += reward
            state = next_state
            prev_coverage = current_coverage

            # 更新策略
            if step % 5 == 0:
                update_info = agent.update(batch_size, current_episode=episode)
                history['actor_losses'].append(update_info['actor_loss'])
                history['critic_losses'].append(update_info['critic_loss'])
                history['coverage_losses'].append(update_info['coverage_loss'])
                history['entropies'].append(update_info['entropy'])

            if done:
                break

        # 记录episode统计信息
        history['episode_rewards'].append(episode_reward)
        history['coverage_counts'].append(current_coverage)

        # 早停检查
        if episode_reward > best_reward:
            best_reward = episode_reward
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count >= early_stopping_patience:
            print(f"Early stopping at episode {episode}")
            break

        # 打印训练进度
        if (episode + 1) % 5 == 0:
            print(f"\nEpisode {episode + 1}/{episodes}")
            print(f"奖励: {episode_reward:.2f}")
            print(f"覆盖率: {current_coverage}")
            print(f"温度: {agent.temp:.3f}")
            print(f"Actor Loss: {history['actor_losses'][-1]:.4f}")
            print(f"Critic Loss: {history['critic_losses'][-1]:.4f}")
            print(f"Coverage Loss: {history['coverage_losses'][-1]:.4f}")
            print(f"Entropy: {history['entropies'][-1]:.4f}")
            print("-" * 50)

    return history


async def main():
    # 创建测试环境
    test_env = PublicTestEnvironment()
    await test_env.get_test_url('httpbin')

    # 开始训练
    print("开始AP-PPO训练...")
    history = await train_ap_ppo(test_env)
    # 保存模型
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_filename = f'ap_ppo_model_{timestamp}.pt'

    save_data = {
        'model_state_dict': agent.network.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'history': history,
        'temperature': agent.temp
    }

    torch.save(save_data, model_filename)
    print(f"Model saved to {model_filename}")

    # 绘制训练曲线
    try:
        import matplotlib.pyplot as plt

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # 奖励曲线
        ax1.plot(history['episode_rewards'])
        ax1.set_title('Episode Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')

        # 覆盖率曲线
        ax2.plot(history['coverage_counts'])
        ax2.set_title('Coverage Progress')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Coverage Count')

        # 损失曲线
        ax3.plot(history['actor_losses'], label='Actor Loss')
        ax3.plot(history['critic_losses'], label='Critic Loss')
        ax3.plot(history['coverage_losses'], label='Coverage Loss')
        ax3.set_title('Training Losses')
        ax3.set_xlabel('Update')
        ax3.set_ylabel('Loss')
        ax3.legend()

        # 熵和温度变化
        ax4.plot(history['entropies'])
        ax4.set_title('Policy Entropy')
        ax4.set_xlabel('Update')
        ax4.set_ylabel('Entropy')

        plt.tight_layout()
        plt.savefig(f'ap_ppo_training_{timestamp}.png')
        print(f"Training curves saved to ap_ppo_training_{timestamp}.png")
        plt.close()

    except ImportError:
        print("matplotlib not installed, skipping visualization")


if __name__ == "__main__":
    asyncio.run(main())
