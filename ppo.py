# public_ppo.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from collections import deque, Counter
import random
import asyncio
import time
from typing import List, Tuple, Dict
import warnings
from env import PublicTestEnvironment, WebFuzzingEnvironment

warnings.filterwarnings('ignore')  # 忽略SSL警告


class PPONetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
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

        self.network = PPONetwork(state_dim, hidden_dim, action_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.memory = deque(maxlen=max_memory_size)
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma

        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5

    def select_action(self, state: torch.Tensor) -> Tuple[int, float]:
        self.network.eval()
        with torch.no_grad():
            action_probs, _ = self.network(state)
            dist = Categorical(action_probs)
            action = dist.sample()
            action_prob = action_probs[action.item()].item()
        self.network.train()
        return action.item(), action_prob

    def store_transition(self, transition: Tuple):
        self.memory.append(transition)

    def update(self, batch_size: int = 32, epochs: int = 10) -> Dict[str, float]:
        if len(self.memory) < batch_size:
            return {
                'actor_loss': 0,
                'critic_loss': 0,
                'entropy': 0,
                'total_loss': 0
            }

        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor(np.array([t[0] for t in batch]))
        actions = torch.LongTensor(np.array([t[1] for t in batch]))
        rewards = torch.FloatTensor(np.array([t[2] for t in batch]))
        next_states = torch.FloatTensor(np.array([t[3] for t in batch]))
        old_probs = torch.FloatTensor(np.array([t[4] for t in batch]))

        with torch.no_grad():
            _, next_values = self.network(next_states)
            _, values = self.network(states)
            advantages = rewards + self.gamma * next_values.squeeze(-1) - values.squeeze(-1)
            returns = advantages + values.squeeze(-1)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0

        for _ in range(epochs):
            action_probs, values = self.network(states)
            dist = Categorical(action_probs)
            new_probs = dist.log_prob(actions).exp()
            entropy = dist.entropy().mean()

            ratio = new_probs / old_probs
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            critic_loss = nn.MSELoss()(values.squeeze(-1), returns)

            loss = (actor_loss +
                    self.value_loss_coef * critic_loss -
                    self.entropy_coef * entropy)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.optimizer.step()

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy += entropy.item()

        return {
            'actor_loss': total_actor_loss / epochs,
            'critic_loss': total_critic_loss / epochs,
            'entropy': total_entropy / epochs,
            'total_loss': (total_actor_loss + self.value_loss_coef * total_critic_loss) / epochs
        }


async def train_fuzzer(
        test_env: PublicTestEnvironment,
        episodes: int = 50,
        max_steps: int = 25,
        batch_size: int = 16,
        update_frequency: int = 5,
        hidden_dim: int = 64,
        early_stopping_threshold: float = -0.1
) -> Dict[str, List[float]]:
    env = WebFuzzingEnvironment(test_env)
    agent = PPOAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        hidden_dim=hidden_dim
    )

    history = {
        'episode_rewards': [],
        'coverage_counts': [],
        'actor_losses': [],
        'critic_losses': [],
        'entropies': []
    }

    reward_window = deque(maxlen=20)

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            # 选择动作
            action, old_prob = agent.select_action(torch.FloatTensor(state))

            # 执行动作
            next_state, reward, done, info = await env.step(action)

            # 存储转换
            transition = (state, action, reward, next_state, old_prob)
            agent.store_transition(transition)

            episode_reward += reward
            state = next_state

            # 定期更新策略
            if len(agent.memory) >= batch_size and step % update_frequency == 0:
                update_info = agent.update(batch_size)
                history['actor_losses'].append(update_info['actor_loss'])
                history['critic_losses'].append(update_info['critic_loss'])
                history['entropies'].append(update_info['entropy'])

            if done:
                break

        # 记录回合统计信息
        history['episode_rewards'].append(episode_reward)
        history['coverage_counts'].append(len(test_env.coverage_history))
        reward_window.append(episode_reward)

        # 打印训练进度
        if (episode + 1) % 5 == 0:
            avg_reward = sum(reward_window) / len(reward_window)
            print(f"\nEpisode {episode + 1}/{episodes}")
            print(f"平均奖励: {avg_reward:.2f}")
            print(f"覆盖率: {len(test_env.coverage_history)}")
            print(
                f"请求次数: {test_env.request_count}/{test_env.test_environments[test_env.current_env]['max_requests']}")

            if history['actor_losses']:
                print(f"Actor Loss: {history['actor_losses'][-1]:.4f}")
                print(f"Critic Loss: {history['critic_losses'][-1]:.4f}")
                print(f"Entropy: {history['entropies'][-1]:.4f}")
            print("-" * 50)

            # 提前停止检查
            if (len(reward_window) == reward_window.maxlen and
                    avg_reward < early_stopping_threshold):
                print("Early stopping due to poor performance")
                break

    return history


async def test_fuzzer(
        test_env: PublicTestEnvironment,
        agent: PPOAgent,
        num_episodes: int = 5,
        max_steps: int = 20
) -> Dict[str, List[float]]:
    env = WebFuzzingEnvironment(test_env)

    test_results = {
        'episode_rewards': [],
        'coverage_counts': [],
        'response_codes': []
    }

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_responses = []

        for step in range(max_steps):
            action, _ = agent.select_action(torch.FloatTensor(state))
            next_state, reward, done, info = await env.step(action)

            episode_reward += reward
            state = next_state

            if info.get('response_status'):
                episode_responses.append(info['response_status'])

            if done:
                break

        test_results['episode_rewards'].append(episode_reward)
        test_results['coverage_counts'].append(len(test_env.coverage_history))
        test_results['response_codes'].extend(episode_responses)

        print(f"\nTest Episode {episode + 1}/{num_episodes}")
        print(f"奖励: {episode_reward:.2f}")
        print(f"覆盖率: {len(test_env.coverage_history)}")
        print(f"响应状态码分布: {Counter(episode_responses)}")

    return test_results


def save_model(agent: PPOAgent, history: Dict, filename: str):
    save_data = {
        'model_state_dict': agent.network.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'history': history
    }
    torch.save(save_data, filename)
    print(f"Model saved to {filename}")


def load_model(filename: str, state_dim: int, action_dim: int, hidden_dim: int = 64) -> PPOAgent:
    agent = PPOAgent(state_dim, action_dim, hidden_dim)
    checkpoint = torch.load(filename)
    agent.network.load_state_dict(checkpoint['model_state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return agent


def plot_training_curves(history: Dict, timestamp: str):
    """绘制训练曲线"""
    try:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

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
        if history['actor_losses']:
            ax3.plot(history['actor_losses'], label='Actor Loss')
            ax3.plot(history['critic_losses'], label='Critic Loss')
            ax3.set_title('Training Losses')
            ax3.set_xlabel('Update')
            ax3.set_ylabel('Loss')
            ax3.legend()

        plt.tight_layout()
        plt.savefig(f'training_curves_{timestamp}.png')
        plt.close()

    except ImportError:
        print("matplotlib not installed, skipping visualization")


async def main():
    # 创建测试环境
    test_env = PublicTestEnvironment()
    await test_env.get_test_url('httpbin')

    # 训练参数
    TRAINING_CONFIG = {
        'episodes': 50,
        'max_steps': 25,
        'batch_size': 16,
        'update_frequency': 5,
        'hidden_dim': 64,
        'early_stopping_threshold': -0.1
    }

    print("开始训练...")
    history = await train_fuzzer(test_env, **TRAINING_CONFIG)

    # 保存模型
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_filename = f'fuzzer_model_{timestamp}.pt'

    env = WebFuzzingEnvironment(test_env)
    agent = PPOAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        hidden_dim=TRAINING_CONFIG['hidden_dim']
    )

    save_model(agent, history, model_filename)

    # 测试模型
    print("\n开始测试...")
    test_results = await test_fuzzer(test_env, agent)

    # 打印最终结果
    print("\n训练完成!")
    print(f"最终覆盖率: {len(test_env.coverage_history)}")
    print(f"总请求数: {test_env.request_count}")
    print(f"平均奖励: {sum(history['episode_rewards']) / len(history['episode_rewards']):.2f}")

    # 绘制训练曲线
    plot_training_curves(history, timestamp)


if __name__ == "__main__":
    asyncio.run(main())
