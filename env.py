# env.py
import requests
import random
import torch
import time
from typing import Dict, List, Tuple, Set, Optional
from collections import deque


class PublicTestEnvironment:
    """公开Web测试环境"""

    def __init__(self):
        self.test_environments = {
            'httpbin': {
                'base_url': 'http://httpbin.org',
                'setup_required': False,
                'rate_limit': 1.0,
                'max_requests': 100
            }
        }

        self.safe_endpoints = {
            'httpbin': [
                '/get',
                '/post',
                '/headers',
                '/cookies',
                '/anything',
                '/status/200'
            ]
        }

        self.current_env = None
        self.request_count = 0
        self.last_request_time = 0
        self.coverage_history = set()

    def respect_rate_limit(self):
        """遵守速率限制"""
        if self.current_env:
            rate_limit = self.test_environments[self.current_env]['rate_limit']
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < rate_limit:
                time.sleep(rate_limit - time_since_last)
            self.last_request_time = time.time()

    async def get_test_url(self, env_name: str = 'httpbin') -> Optional[str]:
        """获取测试URL"""
        if env_name not in self.test_environments:
            print(f"不支持的测试环境: {env_name}")
            return None

        self.current_env = env_name
        return self.test_environments[env_name]['base_url']

    def get_safe_endpoints(self) -> List[str]:
        """获取安全的测试端点"""
        if not self.current_env:
            return []
        return self.safe_endpoints.get(self.current_env, [])

    def can_make_request(self) -> bool:
        """检查是否可以发送请求"""
        if not self.current_env:
            return False
        max_requests = self.test_environments[self.current_env]['max_requests']
        return self.request_count < max_requests

    async def make_request(self, method: str, url: str, params: dict = None,
                           headers: dict = None, data: dict = None) -> Optional[requests.Response]:
        """发送HTTP请求"""
        if not self.can_make_request():
            return None

        self.respect_rate_limit()

        try:
            default_headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'text/html,application/json',
                'Connection': 'keep-alive'
            }
            if headers:
                default_headers.update(headers)

            response = requests.request(
                method=method,
                url=url,
                params=params,
                headers=default_headers,
                data=data,
                timeout=10,
                verify=False,
                allow_redirects=True
            )

            self.request_count += 1

            # 记录覆盖率信息
            coverage_key = f"{response.status_code}_{len(response.content)}"
            self.coverage_history.add(coverage_key)

            return response

        except Exception as e:
            print(f"请求失败: {str(e)}")
            return None


class WebFuzzingEnvironment:
    """Web Fuzzing环境"""

    def __init__(self, test_env: PublicTestEnvironment):
        self.test_env = test_env
        self.current_state = self.get_initial_state()
        self.state_dim = 7  # 状态向量维度
        self.action_dim = 6  # 动作空间维度

        self.mutation_types = [
            'change_param_value',
            'add_param',
            'remove_param',
            'change_method',
            'add_header',
            'modify_content_type'
        ]

    def get_initial_state(self) -> Dict:
        """获取初始状态"""
        return {
            'method': 'GET',
            'params': {},
            'headers': {
                'Content-Type': 'application/json',
                'User-Agent': 'Mozilla/5.0'
            },
            'body': ''
        }

    def reset(self) -> torch.FloatTensor:
        """重置环境"""
        self.current_state = self.get_initial_state()
        return self.state_to_vector(self.current_state)

    def state_to_vector(self, state: Dict) -> torch.FloatTensor:
        """状态向量化"""
        method_encoding = [1 if state['method'] == m else 0
                           for m in ['GET', 'POST', 'PUT', 'DELETE']]
        param_count = len(state['params'])
        header_count = len(state['headers'])
        has_body = 1 if state['body'] else 0

        return torch.FloatTensor(method_encoding + [param_count, header_count, has_body])

    async def step(self, action: int) -> Tuple[torch.FloatTensor, float, bool, Dict]:
        """执行一步动作"""
        mutation_type = self.mutation_types[action]
        new_state = self.mutate_state(self.current_state, mutation_type)

        endpoint = random.choice(self.test_env.get_safe_endpoints())
        base_url = await self.test_env.get_test_url()

        if not base_url:
            return self.state_to_vector(new_state), -1, True, {}

        url = f"{base_url}{endpoint}"

        response = await self.test_env.make_request(
            method=new_state['method'],
            url=url,
            params=new_state['params'],
            headers=new_state['headers'],
            data=new_state['body']
        )

        reward = self.calculate_reward(response)
        done = not self.test_env.can_make_request()

        self.current_state = new_state
        info = {
            'coverage_count': len(self.test_env.coverage_history),
            'response_status': response.status_code if response else None,
            'mutation_type': mutation_type
        }

        return self.state_to_vector(new_state), reward, done, info

    def mutate_state(self, state: Dict, mutation_type: str) -> Dict:
        """状态变异"""
        new_state = state.copy()

        if mutation_type == 'change_param_value':
            if new_state['params']:
                param = random.choice(list(new_state['params'].keys()))
                new_state['params'][param] = str(random.randint(1, 1000))

        elif mutation_type == 'add_param':
            new_param = f"param_{random.randint(1, 1000)}"
            new_state['params'][new_param] = str(random.randint(1, 1000))

        elif mutation_type == 'remove_param':
            if new_state['params']:
                param = random.choice(list(new_state['params'].keys()))
                del new_state['params'][param]

        elif mutation_type == 'change_method':
            methods = ['GET', 'POST']
            new_state['method'] = random.choice(methods)

        elif mutation_type == 'add_header':
            safe_headers = ['Accept', 'Accept-Language', 'Accept-Encoding']
            new_header = random.choice(safe_headers)
            new_state['headers'][new_header] = 'test-value'

        elif mutation_type == 'modify_content_type':
            content_types = [
                'application/json',
                'application/x-www-form-urlencoded',
                'text/plain'
            ]
            new_state['headers']['Content-Type'] = random.choice(content_types)

        return new_state

    def calculate_reward(self, response: Optional[requests.Response]) -> float:
        """计算奖励"""
        if not response:
            return -1

        reward = 0

        # 基础奖励
        if 200 <= response.status_code < 300:
            reward += 1
        elif 400 <= response.status_code < 500:
            reward += 0.5

        # 覆盖率奖励
        coverage_key = f"{response.status_code}_{len(response.content)}"
        if coverage_key not in self.test_env.coverage_history:
            reward += 5

        return reward