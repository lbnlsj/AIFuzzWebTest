# hpl_ppo.py
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


class PatternMemory:
    """攻击模式记忆模块"""

    def __init__(self, max_patterns: int = 100):
        self.patterns = defaultdict(list)  # 按类型存储成功的攻击模式
        self.pattern_scores = defaultdict(float)  # 模式得分
        self.max_patterns = max_patterns

        # 预定义的攻击类型
        self.attack_types = [
            'parameter_fuzzing',
            'header_manipulation',
            'method_switching',
            'content_type_variation'
        ]

    def add_pattern(self, attack_type: str, pattern: Dict, score: float):
        """添加新的攻击模式"""
        if len(self.patterns[attack_type]) >= self.max_patterns:
            # 移除最低分的模式
            min_score_idx = np.argmin([self.pattern_scores[p] for p in self.patterns[attack_type]])
            self.patterns[attack_type].pop(min_score_idx)

        pattern_key = str(pattern)
        self.patterns[attack_type].append(pattern)
        self.pattern_scores[pattern_key] = score

    def get_best_patterns(self, attack_type: str, n: int = 5) -> List[Dict]:
        """获取得分最高的n个模式"""
        patterns = self.patterns[attack_type]
        if not patterns:
            return []

        scores = [self.pattern_scores[str(p)] for p in patterns]
        indices = np.argsort(scores)[-n:]
        return [patterns[i] for i in indices]

    def update_pattern_score(self, pattern: Dict, reward: float):
        """更新模式得分"""
        pattern_key = str(pattern)
        old_score = self.pattern_scores[pattern_key]
        self.pattern_scores[pattern_key] = 0.9 * old_score + 0.1 * reward


class MetaController(nn.Module):
    """元控制器网络"""

    def __init__(self, state_dim: int, hidden_dim: int, num_attack_types: int):
        super(MetaController, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_attack_types),
            nn.Softmax(dim=-1)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class PatternGenerator(nn.Module):
    """模式生成网络"""

    def __init__(self, state_dim: int, hidden_dim: int, pattern_dim: int):
        super(PatternGenerator, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim + pattern_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, pattern_dim),
            nn.Tanh()  # 输出范围[-1,1]
        )

    def forward(self, state: torch.Tensor, pattern: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, pattern], dim=-1)
        return self.network(x)


class HPLPPO:
    """分层模式学习PPO算法"""

    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            pattern_dim: int,
            hidden_dim: int = 64,
            lr: float = 0.001,
            gamma: float = 0.99,
            clip_epsilon: float = 0.2,
            max_memory_size: int = 10000
    ):
        self.pattern_memory = PatternMemory()
        self.meta_controller = MetaController(
            state_dim, hidden_dim, len(self.pattern_memory.attack_types)
        )
        self.pattern_generator = PatternGenerator(state_dim, hidden_dim, pattern_dim)

        # Optimizers
        self.meta_optimizer = optim.Adam(self.meta_controller.parameters(), lr=lr)
        self.generator_optimizer = optim.Adam(self.pattern_generator.parameters(), lr=lr)

        self.memory = deque(maxlen=max_memory_size)
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma

        # 状态标准化
        self.state_mean = None
        self.state_std = None

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """状态标准化"""
        if self.state_mean is None:
            self.state_mean = np.mean(state, axis=0)
            self.state_std = np.std(state, axis=0) + 1e-8

        return (state - self.state_mean) / self.state_std

    def select_attack_type(self, state: torch.Tensor) -> Tuple[str, float]:
        """选择攻击类型"""
        self.meta_controller.eval()
        with torch.no_grad():
            probs = self.meta_controller(state)
            dist = Categorical(probs)
            attack_type_idx = dist.sample()
            attack_type = self.pattern_memory.attack_types[attack_type_idx.item()]
            prob = probs[0, attack_type_idx.item()].item()
        self.meta_controller.train()
        return attack_type, prob

    def generate_pattern(self, state: torch.Tensor,
                         attack_type: str) -> Tuple[Dict, float]:
        """生成攻击模式"""
        self.pattern_generator.eval()
        with torch.no_grad():
            # 获取历史成功模式
            best_patterns = self.pattern_memory.get_best_patterns(attack_type)
            if best_patterns and random.random() < 0.8:  # 80%概率使用历史模式
                base_pattern = random.choice(best_patterns)
                pattern_tensor = torch.FloatTensor(self.dict_to_tensor(base_pattern))
            else:
                pattern_tensor = torch.randn(state.size(0), self.pattern_generator.pattern_dim)

            # 生成新模式
            pattern_tensor = self.pattern_generator(state, pattern_tensor)
            pattern = self.tensor_to_dict(pattern_tensor[0].numpy())
            confidence = torch.mean(torch.abs(pattern_tensor)).item()

        self.pattern_generator.train()
        return pattern, confidence

    def dict_to_tensor(self, pattern: Dict) -> np.ndarray:
        """将模式字典转换为张量"""
        # 简化的转换方法，实际应用中需要更复杂的编码
        return np.array(list(pattern.values()))

    def tensor_to_dict(self, pattern: np.ndarray) -> Dict:
        """将张量转换为模式字典"""
        # 简化的转换方法，实际应用中需要更复杂的解码
        return {
            'param_values': pattern[0],
            'header_values': pattern[1],
            'method_index': pattern[2],
            'content_type_index': pattern[3]
        }

    def store_transition(self, transition: Tuple):
        """存储转换"""
        self.memory.append(transition)

    def update(self, batch_size: int = 32) -> Dict[str, float]:
        """更新策略"""
        if len(self.memory) < batch_size:
            return {
                'meta_loss': 0,
                'generator_loss': 0,
                'total_loss': 0
            }

        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([self.normalize_state(t[0]) for t in batch])
        attack_types = [t[1] for t in batch]
        patterns = [t[2] for t in batch]
        rewards = torch.FloatTensor([t[3] for t in batch])

        # 更新元控制器
        meta_probs = self.meta_controller(states)
        attack_type_indices = torch.LongTensor([
            self.pattern_memory.attack_types.index(at) for at in attack_types
        ])
        meta_loss = -torch.mean(
            torch.log(meta_probs[range(batch_size), attack_type_indices]) * rewards
        )

        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()

        # 更新模式生成器
        pattern_tensors = torch.FloatTensor([
            self.dict_to_tensor(p) for p in patterns
        ])
        generated_patterns = self.pattern_generator(states, pattern_tensors)

        # 使用MSE损失和奖励加权
        generator_loss = torch.mean(
            torch.sum((generated_patterns - pattern_tensors) ** 2, dim=1) * rewards
        )

        self.generator_optimizer.zero_grad()
        generator_loss.backward()
        self.generator_optimizer.step()

        # 更新模式记忆
        for attack_type, pattern, reward in zip(attack_types, patterns, rewards):
            if reward > 0:  # 只存储成功的模式
                self.pattern_memory.add_pattern(attack_type, pattern, reward.item())

        return {
            'meta_loss': meta_loss.item(),
            'generator_loss': generator_loss.item(),
            'total_loss': meta_loss.item() + generator_loss.item()
        }


async def train_hpl_ppo(
        test_env: PublicTestEnvironment,
        episodes: int = 50,
        max_steps: int = 25,
        batch_size: int = 16,
        pattern_dim: int = 4,
        hidden_dim: int = 64,
        early_stopping_patience: int = 20
) -> Dict[str, List[float]]:
    """训练HPL-PPO"""

    env = WebFuzzingEnvironment(test_env)
    agent = HPLPPO(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        pattern_dim=pattern_dim,
        hidden_dim=hidden_dim
    )

    history = {
        'episode_rewards': [],
        'coverage_counts': [],
        'meta_losses': [],
        'generator_losses': [],
        'pattern_counts': []
    }

    best_reward = float('-inf')
    no_improvement_count = 0

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            # 选择攻击类型和生成模式
            state_tensor = torch.FloatTensor([agent.normalize_state(state)])
            attack_type, type_prob = agent.select_attack_type(state_tensor)
            pattern, confidence = agent.generate_pattern(state_tensor, attack_type)

            # 将模式转换为动作
            action = env.pattern_to_action(pattern)  # 需要在WebFuzzingEnvironment中实现

            # 执行动作
            next_state, reward, done, info = await env.step(action)

            # 存储转换
            transition = (state, attack_type, pattern, reward)
            agent.store_transition(transition)

            episode_reward += reward
            state = next_state

            # 更新策略
            if step % 5 == 0:
                update_info = agent.update(batch_size)
                history['meta_losses'].append(update_info['meta_loss'])
                history['generator_losses'].append(update_info['generator_loss'])

            if done:
                break

        # 记录episode统计信息
        history['episode_rewards'].append(episode_reward)
        history['coverage_counts'].append(len(test_env.coverage_history))
        history['pattern_counts'].append(sum(
            len(patterns) for patterns in agent.pattern_memory.patterns.values()
        ))

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
            print(f"覆盖率: {len(test_env.coverage_history)}")
            print(f"学习到的模式数: {history['pattern_counts'][-1]}")
            print(f"Meta Loss: {history['meta_losses'][-1]:.4f}")
            print(f"Generator Loss: {history['generator_losses'][-1]:.4f}")

            # 打印每种攻击类型的模式统计
            print("\n攻击模式统计:")
            for attack_type in agent.pattern_memory.attack_types:
                patterns = agent.pattern_memory.patterns[attack_type]
                if patterns:
                    avg_score = np.mean([
                        agent.pattern_memory.pattern_scores[str(p)]
                        for p in patterns
                    ])
                    print(f"{attack_type}: {len(patterns)} 个模式, 平均分数: {avg_score:.2f}")
            print("-" * 50)

    return history


def plot_training_curves(history: Dict, timestamp: str):
    """绘制训练曲线"""
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
        ax3.plot(history['meta_losses'], label='Meta Loss')
        ax3.plot(history['generator_losses'], label='Generator Loss')
        ax3.set_title('Training Losses')
        ax3.set_xlabel('Update')
        ax3.set_ylabel('Loss')
        ax3.legend()

        # 模式数量变化
        ax4.plot(history['pattern_counts'])
        ax4.set_title('Number of Learned Patterns')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Pattern Count')

        plt.tight_layout()
        plt.savefig(f'hpl_ppo_training_{timestamp}.png')
        print(f"Training curves saved to hpl_ppo_training_{timestamp}.png")
        plt.close()

    except ImportError:
        print("matplotlib not installed, skipping visualization")


class PatternAnalyzer:
    """模式分析器"""

    def __init__(self):
        self.patterns_by_effect = defaultdict(list)
        self.pattern_clusters = defaultdict(list)

    def analyze_pattern(self, pattern: Dict, response: Dict) -> str:
        """分析模式效果"""
        effect_type = 'unknown'
        if response.get('status_code', 0) >= 500:
            effect_type = 'server_error'
        elif response.get('status_code', 0) >= 400:
            effect_type = 'client_error'
        elif 'error' in str(response.get('content', '')).lower():
            effect_type = 'error_message'
        elif response.get('status_code', 0) == 200:
            effect_type = 'success'

        self.patterns_by_effect[effect_type].append(pattern)
        return effect_type

    def cluster_patterns(self, patterns: List[Dict], n_clusters: int = 5):
        """聚类相似模式"""
        if not patterns:
            return

        # 将模式转换为特征向量
        features = []
        for pattern in patterns:
            feature = [
                pattern.get('param_values', 0),
                pattern.get('header_values', 0),
                pattern.get('method_index', 0),
                pattern.get('content_type_index', 0)
            ]
            features.append(feature)

        features = np.array(features)

        # 使用K-means聚类
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=min(n_clusters, len(patterns)))
        clusters = kmeans.fit_predict(features)

        # 存储聚类结果
        for pattern, cluster_id in zip(patterns, clusters):
            self.pattern_clusters[cluster_id].append(pattern)

    def get_pattern_stats(self) -> Dict:
        """获取模式统计信息"""
        stats = {
            'effect_distribution': {
                effect: len(patterns)
                for effect, patterns in self.patterns_by_effect.items()
            },
            'cluster_sizes': {
                cluster_id: len(patterns)
                for cluster_id, patterns in self.pattern_clusters.items()
            },
            'total_patterns': sum(
                len(patterns) for patterns in self.patterns_by_effect.values()
            )
        }
        return stats


async def main():
    # 创建测试环境
    test_env = PublicTestEnvironment()
    await test_env.get_test_url('httpbin')

    # 训练参数
    TRAINING_CONFIG = {
        'episodes': 50,
        'max_steps': 25,
        'batch_size': 16,
        'pattern_dim': 4,
        'hidden_dim': 64,
        'early_stopping_patience': 20
    }

    # 开始训练
    print("开始HPL-PPO训练...")
    history = await train_hpl_ppo(test_env, **TRAINING_CONFIG)

    # 保存模型和训练历史
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_filename = f'hpl_ppo_model_{timestamp}.pt'

    save_data = {
        'meta_controller_state_dict': agent.meta_controller.state_dict(),
        'pattern_generator_state_dict': agent.pattern_generator.state_dict(),
        'pattern_memory': agent.pattern_memory.patterns,
        'pattern_scores': agent.pattern_memory.pattern_scores,
        'history': history
    }

    torch.save(save_data, model_filename)
    print(f"Model saved to {model_filename}")

    # 绘制训练曲线
    plot_training_curves(history, timestamp)

    # 分析学习到的模式
    analyzer = PatternAnalyzer()
    print("\n分析模式效果...")

    for attack_type, patterns in agent.pattern_memory.patterns.items():
        if patterns:
            print(f"\n{attack_type} 模式分析:")
            analyzer.cluster_patterns(patterns)
            stats = analyzer.get_pattern_stats()

            print(f"总模式数: {stats['total_patterns']}")
            print("效果分布:")
            for effect, count in stats['effect_distribution'].items():
                print(f"- {effect}: {count}")

            print("模式聚类:")
            for cluster_id, size in stats['cluster_sizes'].items():
                print(f"- 聚类 {cluster_id}: {size} 个模式")


def update_env(env: WebFuzzingEnvironment):
    """更新环境以支持模式到动作的转换"""

    def pattern_to_action(self, pattern: Dict) -> int:
        """将模式转换为具体动作"""
        # 根据模式中的各个值选择最匹配的动作
        if pattern['method_index'] > 0.5:
            return self.mutation_types.index('change_method')
        elif pattern['content_type_index'] > 0.5:
            return self.mutation_types.index('modify_content_type')
        elif pattern['header_values'] > 0.5:
            return self.mutation_types.index('add_header')
        else:
            if pattern['param_values'] > 0:
                return self.mutation_types.index('add_param')
            else:
                return self.mutation_types.index('change_param_value')

    # 动态添加方法
    setattr(WebFuzzingEnvironment, 'pattern_to_action', pattern_to_action)


if __name__ == "__main__":
    # 更新环境
    update_env(WebFuzzingEnvironment)
    # 运行主程序
    asyncio.run(main())
