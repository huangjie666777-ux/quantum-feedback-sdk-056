# Quantum Feedback SDK

一个可嵌入的小型量子电路模拟 SDK，使用密度矩阵确定性处理测量、经典反馈、reset 和相位翻转噪声，不使用随机轨迹近似。

## 安装与接口

开发环境使用：

```bash
.venv/bin/python -m pip wheel --no-build-isolation --no-deps . -w dist
```

主入口是 `run(num_qubits, num_clbits, operations)`，其中量子位数和经典位数都必须是 1 到 6 的整数，所有量子位和经典位初始为 0。`operations` 为按时间顺序执行的映射列表：

- `{"op": "H" | "X" | "Z", "qubit": q}`：单量子位门。
- `{"op": "RZ", "qubit": q, "angle": theta}`：角度单位为弧度。
- `{"op": "CX", "control": c, "target": t}`：控制位与目标位不能相同。
- `{"op": "MEASURE", "qubit": q, "clbit": c}`：计算基测量并覆写经典位。
- `{"op": "RESET", "qubit": q}`：将该量子位非破坏性地重置为 0，保留其他量子位的约化状态，不修改经典位。
- `{"op": "PHASE_FLIP", "qubit": q, "p": p}`：确定性混合信道 `(1-p)ρ + p ZρZ`，`0 <= p <= 1`。
- 任意门可加 `"condition": [c, v]`：只有经典位 `c` 等于 `v`（0 或 1）时执行。

返回的 `ExecutionResult` 提供：

- `density_matrix`：末态量子密度矩阵副本。
- `classical_probabilities`：经典位串到概率的映射。
- `samples(shots, seed)`：使用局部 `numpy.random.Generator` 采样，不污染全局 NumPy 随机状态。
- `sample_counts(shots, seed)`：返回位串计数。

## 位序

量子位 0 对应计算基整数索引的最低位。例如 2 量子位 Bell 态是 `|00> + |11>`，其基态索引为 0 和 3。经典输出位串高位在左：经典位 0 的值显示在最右侧，未测量经典位保持 0，因此 3 个经典位中只有 `c0=1` 时输出 `001`。

测量后，模拟器保留带概率的归一化条件密度矩阵分支。后续条件门只作用于匹配的经典分支；不同读数分支不发生相干干涉。只有经典记录完全相同的分支才按概率混合。经典位被覆写后，相同新读数的分支也会按概率混合。

## 校验

越界量子位或经典位、相同的 CX 控制位和目标位、未知操作、非有限 RZ 角度、非法概率和非法反馈条件都会抛出 `CircuitValidationError`。错误消息包含从 0 开始的操作位置，例如 `operation 3`。校验在分配量子态前完成；SDK 不会返回部分执行结果，也不会修改调用方传入的数据。

## 数值容差

所有矩阵使用 NumPy `complex128`。内部比较用于剪除严格为 0 的概率分支；测试默认使用 `1e-10` 的绝对和相对容差。最多 6 个量子位时密度矩阵为 64×64，确定性演化开销可控。

## 测试与示例

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
PYTHONPATH=src .venv/bin/python examples/feedback_noise.py
```

示例包含 Bell 纠缠、相位翻转噪声、测量以及基于测量结果的条件 X 门。
