# ap_ppo.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from collections import deque, defaultdict
import random
from typing import List, Tuple, Dict, Optional
from env import WebFuzzingEnvironment
import matplotlib.pyplot as plt
import time


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
            if indices:  # 确保episode有数据
                episode_priorities[ep_id] = np.mean(priorities[indices])

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
                self.priorities[idx] = priority + 1e-6  # 添加小值防止优先级为0


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

        # 初始化优先级回放缓冲区
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
        # 综合考虑奖励和覆盖率增益
        return abs(reward) + self.coverage_weight * coverage_gain

    def update(self,
               batch_size: int = 32,
               epochs: int = 10,
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
            ratio = new_probs / old_probs

            surr1 = ratio * advantages * weights
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages * weights
            actor_loss = -torch.min(surr1, surr2).mean()

            # 价值损失
            critic_loss = nn.MSELoss()(values.squeeze(-1), returns) * weights.mean()

            # 覆盖率预测损失
            coverage_loss = nn.BCELoss()(coverage_preds.squeeze(-1), coverage_gains)

            # 熵正则化
            entropy = dist.entropy().mean()

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

        # 计算平均损失
        avg_actor_loss = total_actor_loss / epochs
        avg_critic_loss = total_critic_loss / epochs
        avg_coverage_loss = total_coverage_loss / epochs
        avg_entropy = total_entropy / epochs

        return {
            'actor_loss': avg_actor_loss,
            'critic_loss': avg_critic_loss,
            'coverage_loss': avg_coverage_loss,
            'entropy': avg_entropy,
            'total_loss': avg_actor_loss + 0.5 * avg_critic_loss + self.coverage_weight * avg_coverage_loss
        }


def train_ap_ppo(
        target_url: str,
        episodes: int = 1000,
        max_steps: int = 100,
        batch_size: int = 32,
        hidden_dim: int = 64,
        early_stopping_patience: int = 50
) -> Dict[str, List[float]]:
    """训练AP-PPO"""
    env = WebFuzzingEnvironment(target_url)
    agent = AdaptivePriorityPPO(
        state_dim=env.observation_space_dim,
        action_dim=env.action_space_dim,
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
        prev_coverage = len(env.coverage_history)

        for step in range(max_steps):
            # 选择动作
            action, action_prob, coverage_pred = agent.select_action(torch.FloatTensor(state))

            # 执行动作
            next_state, reward, done, info = env.step(action)

            # 计算覆盖率增益
            current_coverage = len(env.coverage_history)
            coverage_gain = float(current_coverage > prev_coverage)

            # 计算优先级并存储转换
            priority = agent.calculate_priority(reward, coverage_gain)
            transition = (state, action, reward, next_state, action_prob, coverage_gain)
            agent.memory.add(transition, priority, episode)

            episode_reward += reward
            state = next_state
            prev_coverage = current_coverage

            # 更新策略
            if step % 10 == 0:
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
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}")
            print(f"Reward: {episode_reward:.2f}")
            print(f"Coverage: {current_coverage}")

            print(f"Temperature: {agent.temp:.3f}")
            print(f"Actor Loss: {history['actor_losses'][-1]:.4f}")
            print(f"Critic Loss: {history['critic_losses'][-1]:.4f}")
            print(f"Coverage Loss: {history['coverage_losses'][-1]:.4f}")
            print(f"Entropy: {history['entropies'][-1]:.4f}")
            print("-" * 50)

        return history


class APPOMetrics:
    """AP-PPO指标跟踪和分析"""

    def __init__(self):
        self.coverage_history = []
        self.reward_history = []
        self.temp_history = []
        self.success_rate_history = []
        self.priority_stats = []

    def update(self,
               coverage: int,
               reward: float,
               temp: float,
               success_rate: float,
               priorities: List[float]):
        """更新指标"""
        self.coverage_history.append(coverage)
        self.reward_history.append(reward)
        self.temp_history.append(temp)
        self.success_rate_history.append(success_rate)
        self.priority_stats.append({
            'mean': np.mean(priorities),
            'std': np.std(priorities),
            'max': np.max(priorities),
            'min': np.min(priorities)
        })

    def get_statistics(self) -> Dict[str, float]:
        """获取统计信息"""
        return {
            'avg_coverage': np.mean(self.coverage_history),
            'max_coverage': np.max(self.coverage_history),
            'avg_reward': np.mean(self.reward_history),
            'avg_temp': np.mean(self.temp_history),
            'avg_success_rate': np.mean(self.success_rate_history),
            'priority_mean': np.mean([s['mean'] for s in self.priority_stats]),
            'priority_std': np.mean([s['std'] for s in self.priority_stats])
        }

    def plot_metrics(self):
        """绘制指标图表"""

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 覆盖率变化
        axes[0, 0].plot(self.coverage_history)
        axes[0, 0].set_title('Coverage Progress')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Coverage Count')

        # 奖励变化
        axes[0, 1].plot(self.reward_history)
        axes[0, 1].set_title('Reward Progress')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Episode Reward')

        # 温度参数变化
        axes[1, 0].plot(self.temp_history)
        axes[1, 0].set_title('Temperature Parameter')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Temperature')

        # 优先级统计
        priority_means = [s['mean'] for s in self.priority_stats]
        priority_stds = [s['std'] for s in self.priority_stats]
        axes[1, 1].plot(priority_means, label='Mean Priority')
        axes[1, 1].fill_between(
            range(len(priority_means)),
            np.array(priority_means) - np.array(priority_stds),
            np.array(priority_means) + np.array(priority_stds),
            alpha=0.3
        )
        axes[1, 1].set_title('Priority Statistics')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Priority')
        axes[1, 1].legend()

        plt.tight_layout()
        return fig


def main():
    """主函数"""
    # 训练配置
    CONFIG = {
        'target_url': "http://example.com/api",  # 替换为实际目标
        'episodes': 1000,
        'max_steps': 100,
        'batch_size': 32,
        'hidden_dim': 64,
        'early_stopping_patience': 50,
    }

    # 初始化环境
    env = WebFuzzingEnvironment(CONFIG['target_url'])

    # 初始化智能体
    agent = AdaptivePriorityPPO(
        state_dim=env.observation_space_dim,
        action_dim=env.action_space_dim,
        hidden_dim=CONFIG['hidden_dim']
    )

    # 初始化指标跟踪器
    metrics = APPOMetrics()

    # 训练模型
    print("Starting AP-PPO training...")
    history = train_ap_ppo(
        target_url=CONFIG['target_url'],
        episodes=CONFIG['episodes'],
        max_steps=CONFIG['max_steps'],
        batch_size=CONFIG['batch_size'],
        hidden_dim=CONFIG['hidden_dim'],
        early_stopping_patience=CONFIG['early_stopping_patience']
    )

    # 更新训练过程中的指标
    for episode in range(len(history['episode_rewards'])):
        metrics.update(
            coverage=history['coverage_counts'][episode],
            reward=history['episode_rewards'][episode],
            temp=agent.temp,
            success_rate=sum(agent.success_rate) / len(agent.success_rate) if agent.success_rate else 0,
            priorities=list(agent.memory.priorities)
        )

    # 分析结果
    print("\nTraining completed!")
    print("\nFinal Statistics:")
    stats = metrics.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value:.4f}")

    # 绘制结果
    try:
        fig = metrics.plot_metrics()
        plt.show()
    except ImportError:
        print("matplotlib not installed, skipping visualization")

    # 保存模型和指标（可选）
    save_results = input("\nDo you want to save the results? (y/n): ")
    if save_results.lower() == 'y':
        try:
            import time
            timestamp = time.strftime("%Y%m%d-%H%M%S")

            # 保存模型和训练状态
            save_dict = {
                'model_state_dict': agent.network.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'history': history,
                'metrics': metrics.__dict__,
                'config': CONFIG,
                'final_temperature': agent.temp,
                'memory_stats': {
                    'size': len(agent.memory.buffer),
                    'priorities': list(agent.memory.priorities)
                }
            }

            # 保存模型和训练数据
            torch.save(save_dict, f'ap_ppo_model_{timestamp}.pt')

            # 保存可视化结果
            try:
                fig.savefig(f'ap_ppo_metrics_{timestamp}.png')
            except Exception as e:
                print(f"Could not save figure: {e}")

            print(f"Results saved with timestamp: {timestamp}")

            # 打印保存的模型位置
            print(f"Model saved to: ap_ppo_model_{timestamp}.pt")
            print(f"Metrics plot saved to: ap_ppo_metrics_{timestamp}.png")
        except Exception as e:
            print(f"Error saving results: {e}")


def load_and_test_model(model_path: str, target_url: str, num_episodes: int = 10):
    """加载和测试保存的模型"""
    # 加载保存的模型和配置
    checkpoint = torch.load(model_path)
    config = checkpoint['config']

    # 初始化环境和智能体
    env = WebFuzzingEnvironment(target_url)
    agent = AdaptivePriorityPPO(
        state_dim=env.observation_space_dim,
        action_dim=env.action_space_dim,
        hidden_dim=config['hidden_dim']
    )

    # 加载模型参数
    agent.network.load_state_dict(checkpoint['model_state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # 测试模式
    agent.network.eval()
    test_metrics = {
        'rewards': [],
        'coverage': [],
        'success_rate': []
    }

    print(f"\nTesting model for {num_episodes} episodes...")

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        steps = 0
        successes = 0

        while steps < config['max_steps']:
            # 使用模型选择动作（测试模式）
            action, _, _ = agent.select_action(torch.FloatTensor(state), training=False)

            # 执行动作
            next_state, reward, done, info = env.step(action)

            episode_reward += reward
            if reward > 0:
                successes += 1

            state = next_state
            steps += 1

            if done:
                break

        # 记录测试指标
        test_metrics['rewards'].append(episode_reward)
        test_metrics['coverage'].append(len(env.coverage_history))
        test_metrics['success_rate'].append(successes / steps if steps > 0 else 0)

        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"Reward: {episode_reward:.2f}")
        print(f"Coverage: {len(env.coverage_history)}")
        print(f"Success Rate: {test_metrics['success_rate'][-1]:.2%}")
        print("-" * 30)

    # 打印总体测试结果
    print("\nTest Results:")
    print(f"Average Reward: {np.mean(test_metrics['rewards']):.2f}")
    print(f"Average Coverage: {np.mean(test_metrics['coverage']):.2f}")
    print(f"Average Success Rate: {np.mean(test_metrics['success_rate']):.2%}")

    return test_metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='AP-PPO Training and Testing')
    parser.add_argument('--mode', choices=['train', 'test'], default='train',
                        help='Run mode: train or test')
    parser.add_argument('--model-path', type=str, help='Path to saved model for testing')
    parser.add_argument('--target-url', type=str, required=True,
                        help='Target URL for fuzzing')
    parser.add_argument('--test-episodes', type=int, default=10,
                        help='Number of test episodes')

    args = parser.parse_args()

    if args.mode == 'train':
        main()
    else:
        if not args.model_path:
            print("Error: --model-path is required for test mode")
        else:
            load_and_test_model(args.model_path, args.target_url, args.test_episodes)
