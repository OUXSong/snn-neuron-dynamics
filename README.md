# 脉冲神经网络（SNN）神经元动态过程仿真

这是一个简单但完整的 Python 仿真示例，用于理解脉冲神经元的动态过程。它采用经典的 LIF（Leaky Integrate-and-Fire）模型，展示膜电位如何随时间积累、触发放电、并在脉冲后复位。

## 模型简介

LIF 神经元的核心动力学方程：

```text
dV/dt = -(V - V_rest) / tau_m + (R * I) / tau_m
```

当 `V >= V_th` 时：
- 产生一个脉冲（spike）
- 将膜电位重置为 `V_reset`

## 文件说明

- `snn_neuron_simulation.py`：单神经元仿真脚本
- `requirements.txt`：运行依赖

## 运行方式

```bash
pip install -r requirements.txt
python snn_neuron_simulation.py
```

## 结果解释

脚本会生成两张图：

1. 输入电流图：展示外部刺激信号
2. 膜电位与脉冲图：展示神经元如何积分、达到阈值并发放动作电位

## 核心过程

1. 输入电流进入神经元
2. 膜电位逐步累积
3. 当达到发放阈值时，神经元发放 spike
4. spike 后膜电位被重置，进入下一轮积分

## 适用场景

- 学习 SNN 基础原理
- 理解神经元动态过程
- 为更复杂的神经形态网络仿真打基础

如果你想进一步扩展，可以继续加入：
- 多个神经元连接
- STDP 学习规则
- 动画可视化
- 时空编码（rate coding / temporal coding）
