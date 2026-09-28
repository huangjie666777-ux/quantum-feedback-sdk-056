# Quantum feedback SDK

一个可嵌入教学程序的确定性量子电路模拟 SDK，使用密度矩阵精确处理测量、经典反馈、reset 与概率噪声，不使用随机轨迹近似。

## 环境与安装

- Python 3.10，NumPy 2.2.6。
- 本地测试可直接设置 `PYTHONPATH=src`；也可构建并安装 wheel：

```bash
.venv/bin/python -m pip wheel --no-build-isolation --no-deps . -w dist
.venv/bin/python -m pip install --force-reinstall dist/quantum_feedback_sdk-0.1.0-py3-none-any.whl
```

## 接口

```python
from quantum_feedback import run_circuit

result = run_circuit(n_qubits, n_cbits, operations)
result.density_matrix              # 2^n x 2^n numpy.ndarray，返回副本
result.classical_probabilities     # {"010": 概率, ...}
result.sample(shots=100, seed=42)  # 私有 RNG 采样，不改变精确结果或 NumPy 全局随机状态
```

`run_circuit` 会先完整校验电路，再执行模拟；非法输入抛出 `CircuitError`，错误信息包含从 0 开始的操作位置，例如 `operation 3: ...`。操作列表在内部深拷贝，调用方数据不会被修改。量子位和经典位数量均必须为 1 至 6 的整数。未测量经典位保持 `0`。

## 操作

操作按列表顺序执行，角度单位为弧度：

- `{"op": "H", "qubit": q}`
- `{"op": "X", "qubit": q}`
- `{"op": "Z", "qubit": q}`
- `{"op": "RZ", "qubit": q, "angle": theta}`：`diag(e^-iθ/2, e^iθ/2)`
- `{"op": "CX", "control": c, "target": t}`
- `{"op": "measure", "qubit": q, "cbit": b}`：计算基 Born 测量，投影坍缩并覆写经典位
- `{"op": "reset", "qubit": q}`：非选择性 reset 到 `|0>`，保留其余量子位的约化状态，不清空经典位
- `{"op": "phase_flip", "qubit": q, "p": p}`：在当前操作位置施加 `(1-p)ρ + p ZρZ`
- 量子门（含 `CX`）可添加 `"condition": {"cbit": b, "equals": 0或1}`

## 位序与反馈语义

量子基态索引中，量子位 0 是最低有效位：`|q2 q1 q0>` 的索引为 `4q2 + 2q1 + q0`。经典输出字符串高位在左：经典位 0 位于输出串最右侧，经典位越大越靠左。

测量为每个读数保留独立的未归一化密度矩阵分支，分支迹长就是该联合结果概率。后续条件门只作用于满足经典条件的分支。相同经典读数的分支只做概率加权混合，不会重新相干叠加。最终 `density_matrix` 是所有读数分支的非相干求和。

## 数值容差

默认内部结果校验容差为 `quantum_feedback.TOLERANCE = 1e-10`。所有矩阵使用 `numpy.complex128`；返回前对密度矩阵做厄米化 `(ρ + ρ†)/2` 以消除浮点级别非厄米误差。可向 `run_circuit(..., tolerance=...)` 传入自定义概率归一化容差。

## 测试与示例

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
PYTHONPATH=src .venv/bin/python examples/feedback_noise.py
```

示例构造 Bell 态，在指定位置加入相位翻转噪声，测量一个量子位，再根据经典读数条件纠正另一个量子位。
