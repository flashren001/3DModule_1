# stable_baselines3 完整使用指南

## 📋 概述

`stable_baselines3` 是目前最流行的Python强化学习库之一，提供了高质量的强化学习算法实现，包括PPO、SAC、DQN、A2C、TD3等。

## 🚀 安装方法

### 基础安装
```bash
pip install stable-baselines3
```

### 包含额外功能
```bash
pip install stable-baselines3[extra]
```

### 从源码安装最新版本
```bash
pip install git+https://github.com/DLR-RM/stable-baselines3
```

### 验证安装
```bash
python -c "from stable_baselines3 import PPO; print('安装成功！')"
```

## 📦 正确的导入方法

### 基本算法
```python
from stable_baselines3 import PPO, SAC, DQN, A2C, TD3
```

### 环境工具
```python
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
```

### 回调函数
```python
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
```

### 策略网络
```python
from stable_baselines3.common.policies import ActorCriticPolicy
```

### 工具函数
```python
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
```

### 监控和日志
```python
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
```

## 🔧 基本使用示例

### 简单训练
```python
import gymnasium as gym
from stable_baselines3 import PPO

# 1. 创建环境
env = gym.make('CartPole-v1')

# 2. 创建模型
model = PPO('MlpPolicy', env, verbose=1)

# 3. 训练模型
model.learn(total_timesteps=10000)

# 4. 保存模型
model.save("ppo_cartpole")

# 5. 加载模型
model = PPO.load("ppo_cartpole")

# 6. 测试模型
obs, _ = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()
```

### 自定义环境
```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO

class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        
        # 定义动作空间和观察空间
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.np_random.normal(size=(4,))
        return self.state.astype(np.float32), {}
    
    def step(self, action):
        self.state += self.np_random.normal(size=(4,)) * 0.1
        reward = 1.0 if action == 1 else -1.0
        terminated = False
        truncated = False
        info = {}
        return self.state.astype(np.float32), reward, terminated, truncated, info

# 使用自定义环境
env = CustomEnv()
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=5000)
```

## 📞 回调函数使用

```python
from stable_baselines3.common.callbacks import BaseCallback

class CustomCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.best_mean_reward = -np.inf
        
    def _on_training_start(self) -> None:
        print("训练开始")
        
    def _on_rollout_start(self) -> None:
        print("回滚开始")
        
    def _on_step(self) -> bool:
        # 每步调用
        if self.n_calls % 1000 == 0:
            print(f"步数: {self.n_calls}")
        return True  # 继续训练
        
    def _on_rollout_end(self) -> None:
        print("回滚结束")
        
    def _on_training_end(self) -> None:
        print("训练结束")

# 使用回调函数
callback = CustomCallback()
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000, callback=callback)
```

## ⚙️ 超参数调优

### PPO 超参数
```python
model = PPO(
    'MlpPolicy',
    env,
    learning_rate=3e-4,      # 学习率
    n_steps=2048,            # 每次更新的步数
    batch_size=64,           # 批量大小
    n_epochs=10,             # 优化轮数
    gamma=0.99,              # 折扣因子
    gae_lambda=0.95,         # GAE参数
    clip_range=0.2,          # PPO裁剪参数
    ent_coef=0.01,           # 熵系数
    vf_coef=0.5,             # 价值函数系数
    max_grad_norm=0.5,       # 梯度裁剪
    verbose=1,
    device='auto'            # 自动选择设备
)
```

### SAC 超参数
```python
from stable_baselines3 import SAC

model = SAC(
    'MlpPolicy',
    env,
    learning_rate=3e-4,
    buffer_size=1000000,     # 经验回放缓冲区大小
    learning_starts=100,     # 开始学习的步数
    batch_size=256,          # 批量大小
    tau=0.005,               # 软更新参数
    gamma=0.99,              # 折扣因子
    train_freq=1,            # 训练频率
    gradient_steps=1,        # 梯度更新步数
    verbose=1
)
```

## 🐛 常见错误及解决方案

### 1. 导入错误
**错误**: `ImportError: No module named 'stable_baselines3'`

**解决方案**:
```bash
pip install stable-baselines3
# 或者检查虚拟环境是否激活
```

### 2. gymnasium vs gym 冲突
**错误**: `AttributeError: module 'gym' has no attribute 'make'`

**解决方案**:
```bash
# 卸载旧版gym，安装gymnasium
pip uninstall gym
pip install gymnasium

# 代码中使用
import gymnasium as gym
```

### 3. 环境类型错误
**错误**: `ValueError: The environment must inherit from gymnasium.Env`

**解决方案**:
```python
# 确保继承正确的基类
import gymnasium as gym

class MyEnv(gym.Env):  # 继承gymnasium.Env
    def __init__(self):
        super(MyEnv, self).__init__()
        # ...
```

### 4. 动作空间错误
**错误**: `AssertionError: The action space must be of type gymnasium.spaces`

**解决方案**:
```python
from gymnasium import spaces

# 使用gymnasium.spaces定义空间
self.action_space = spaces.Discrete(2)
self.observation_space = spaces.Box(low=-1, high=1, shape=(4,))
```

### 5. CUDA内存不足
**错误**: `CUDA out of memory`

**解决方案**:
```python
# 使用CPU训练
model = PPO('MlpPolicy', env, device='cpu')

# 或减少批量大小
model = PPO('MlpPolicy', env, batch_size=32)
```

## 💡 最佳实践

### 1. 环境验证
```python
from stable_baselines3.common.env_checker import check_env

# 验证自定义环境
check_env(env, warn=True)
```

### 2. 环境标准化
```python
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

# 包装环境
env = DummyVecEnv([lambda: env])
env = VecNormalize(env, norm_obs=True, norm_reward=True)
```

### 3. 并行训练
```python
from stable_baselines3.common.vec_env import SubprocVecEnv

# 创建并行环境
env = SubprocVecEnv([lambda: gym.make('CartPole-v1') for _ in range(4)])
```

### 4. 模型评估
```python
from stable_baselines3.common.evaluation import evaluate_policy

# 评估模型性能
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
print(f"平均奖励: {mean_reward:.2f} ± {std_reward:.2f}")
```

### 5. TensorBoard监控
```python
# 启用TensorBoard日志
model = PPO('MlpPolicy', env, tensorboard_log="./ppo_tensorboard/")

# 查看日志
# tensorboard --logdir ./ppo_tensorboard/
```

### 6. 检查点保存
```python
from stable_baselines3.common.callbacks import CheckpointCallback

# 定期保存模型
checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./logs/')
model.learn(total_timesteps=10000, callback=checkpoint_callback)
```

## 🔄 完整训练流程

```python
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy

# 1. 创建环境
env = make_vec_env('CartPole-v1', n_envs=4)
env = VecNormalize(env, norm_obs=True, norm_reward=True)

# 2. 创建评估环境
eval_env = make_vec_env('CartPole-v1', n_envs=1)
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, training=False)

# 3. 创建回调函数
eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
                            log_path='./logs/', eval_freq=500,
                            deterministic=True, render=False)

checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./logs/')

# 4. 创建模型
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log="./ppo_tensorboard/")

# 5. 训练模型
model.learn(total_timesteps=50000, callback=[eval_callback, checkpoint_callback])

# 6. 保存最终模型
model.save("ppo_final")

# 7. 评估模型
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=100)
print(f"最终评估结果: {mean_reward:.2f} ± {std_reward:.2f}")
```

## 📚 算法选择指南

### PPO (Proximal Policy Optimization)
- **适用于**: 连续和离散动作空间
- **特点**: 稳定、易调参
- **推荐场景**: 通用强化学习任务

### SAC (Soft Actor-Critic)
- **适用于**: 连续动作空间
- **特点**: 样本效率高、探索性好
- **推荐场景**: 机器人控制、连续控制任务

### DQN (Deep Q-Network)
- **适用于**: 离散动作空间
- **特点**: 经典算法、易理解
- **推荐场景**: 游戏、离散决策任务

### A2C (Advantage Actor-Critic)
- **适用于**: 连续和离散动作空间
- **特点**: 简单、快速
- **推荐场景**: 简单任务、原型开发

### TD3 (Twin Delayed DDPG)
- **适用于**: 连续动作空间
- **特点**: 稳定的确定性策略
- **推荐场景**: 连续控制、需要确定性策略的任务

## 🔗 相关资源

- **官方文档**: https://stable-baselines3.readthedocs.io/
- **GitHub仓库**: https://github.com/DLR-RM/stable-baselines3
- **示例代码**: https://github.com/DLR-RM/rl-baselines3-zoo
- **学术论文**: https://jmlr.org/papers/v22/20-1364.html
- **教程视频**: https://www.youtube.com/c/MachineLearningwithPhil

## ✅ 总结

stable_baselines3 是一个功能强大且易用的强化学习库。正确使用它的关键要点：

1. **安装**: 使用 `pip install stable-baselines3`
2. **导入**: 使用完整的模块路径导入
3. **环境**: 使用 gymnasium 而不是旧版 gym
4. **验证**: 使用 check_env 验证自定义环境
5. **监控**: 使用回调函数和TensorBoard监控训练
6. **评估**: 使用 evaluate_policy 评估模型性能

遵循这些最佳实践，您就能够成功使用 stable_baselines3 进行强化学习项目开发。