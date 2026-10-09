# LightGBM 镜像样本分裂搜索问题复现

[English](README.md)

本仓库提供源码消融实验实际使用的 LightGBM 4.6.0 补丁、原编译配置，以及可直接运行的合成数据示例。示例检验的问题是：父节点排除某个特征后，子节点继续沿用这一排除状态，即使其他特征完成分裂后，该特征已能降低误差。补丁取消串行树学习器中的这项继承判断。

所有输入均由 `datasets.py` 中的函数生成，不包含气动数据库。新增连续合成数据用于独立验证上述现象，与论文中的气动实验结果分开报告。

## v1.0.2 修正

单树验证现在要求每个预期的 `(variant, a, seed, condition)` 组合恰好出现一次：`full` 为 80 条，`smoke` 为 16 条。缺失、重复、非预期组合或无效数值，会在机理和归档比较之前判定失败。NRMSE 与根节点增益必须是有限非负实数，分裂计数必须是非负整数，均排除布尔值。每个归档差值分别检查，避免 NaN 被最大值计算忽略；记录不完整或数值无效时，最大差值为 `null`。

23 项单元测试全部通过，覆盖报告中的整组缺失、等量重复补齐、单条记录和 NaN 故障注入。存档 full 结果及其种子 11 的 smoke 子集分别通过 17 项检查。本次复核保存结果，未重新编译或训练；补丁、构建脚本、配置、数据及原始数值结果保持不变。检查记录见 `expected/revalidated_v1.0.2.json`。

直接调用时使用 `verify_results(single_rows, continuous_rows, suite='full')`，快速检查须明确指定 `suite='smoke'`。输入的数值字段须与 worker 输出一样使用数值类型。预期组合由调用者声明，不从不完整记录中推断；命令行控制器会传入 `--suite` 设置。

## v1.0.1 修正

本版完整检查四种连续数据组合，包括此前漏检的“修补版＋独立噪声”。缺失或重复组合、无效指标、任一干预组合未恢复几何分裂、验证误差未低于未扰动基线，均会导致检查失败。构建命令会在抛出异常前保存标准输出、标准错误、退出码以及超时或启动失败的诊断。

14 项单元测试通过，覆盖了漏检组合的故障注入和构建失败路径。原有 80 次单树及 4 次连续数据结果通过更新后的 15 项检查。本次对保存结果进行复核，未重新编译或训练模型；原补丁、编译参数、数据和数值结果保持不变。检查记录见 `expected/revalidated_v1.0.1.json`。

## 文件说明

- `patches/parent_feature_reevaluation.patch`：实验原样补丁。
- `config/build.json`：固定的上游提交及实际执行的 CMake 编译选项。
- `config/original_build.json`、`config/original_*_CMakeCache.txt`：原编译记录；个人机器路径已替换为占位符，其余配置值保留。
- `scripts/build.py`：分别编译原始版与修补版，不替换已安装的 LightGBM。
- `datasets.py`：四单元合成数据、噪声对照和连续合成外形的生成函数。
- `data/`：上述函数生成的 CSV 数据；种子和校验值见 `data/manifest.json`。
- `reproduce.py`：在独立进程中加载两种程序库，检查误差及几何分裂次数。
- `expected/`：原单树实验记录和公开包重新编译后的验证结果。
- `tests/test_datasets.py`：解析增益、对称性、噪声匹配和数据生成的单元测试。
- `tests/test_verification.py`、`tests/test_build_logging.py`：验证组合完整性及构建失败日志的回归测试。

## 编译与运行

已验证环境：Windows x64、Python 3.12.13、GCC 14.1.0、CMake 4.4.3、Ninja 1.13.2、NumPy 2.4.3、SciPy 1.17.1。Windows 下使用支持 OpenMP 的 64 位 MinGW GCC。将 Git、GCC（`gcc` 和 `g++`）、CMake、Ninja 加入 `PATH`。

在仓库目录建立并激活 Python 虚拟环境，然后运行：

```text
python -m venv .venv
```

PowerShell 激活命令为 `.\.venv\Scripts\Activate.ps1`，POSIX shell 为 `source .venv/bin/activate`。继续执行：

```text
python -m pip install -r requirements.txt -r build-requirements.txt
python scripts/build.py
python -m unittest discover -s tests -v
python reproduce.py
```

默认编译过程联网获取固定的上游版本及子模块，核验提交和补丁校验值，再分别编译 CPU 动态库。运行时使用对应版本的 Python 封装和一个训练线程。Windows 下脚本会添加所选编译器的 DLL 搜索目录。

工具不在 `PATH` 时，可通过 `--cmake`、`--ninja`、`--cc`、`--cxx` 指定绝对路径。已有包含完整子模块的 LightGBM 克隆时，可通过 `--source <目录>` 使用；脚本导出固定提交的 Git 对象，不读取该克隆中的工作区修改。`--jobs 4` 设置编译并行度，不改变训练线程数。脚本包含 POSIX 构建支持；本次验证环境为 Windows。

脚本不覆盖已有编译目录或结果目录。再次运行时指定新目录：

```text
python scripts/build.py --work-dir build/second
python reproduce.py --build-manifest build/second/build_manifest.json --output runs/second
```

完整验证包含 80 次单树拟合和 4 次连续数据拟合。快速检查只用一个噪声种子：

```text
python reproduce.py --suite smoke --output runs/smoke
```

单独导出数据而不训练：

```text
python datasets.py --output runs/data_only
```

## 验证内容

原单树实验使用 `y = beta * (a + z)`，其中 `beta` 和 `z` 取正负 1，每种组合重复 128 次。两种程序库各使用 `a = 0, 1`、种子 11–15，以及未扰动、独立高斯噪声、等 RMS 奇对称噪声和正最小分裂增益阈值四种条件。

当 `a = 1` 时，原始版在根节点按侧滑角变量分裂，但没有几何分裂，NRMSE 为 0.707107；修补版恢复两个几何分裂，NRMSE 为零。根节点侧滑角增益为 512，每个子节点的几何增益为 256。当 `a = 0` 时，两种程序库均无法完成初始分裂，NRMSE 为 1。这里的 NRMSE 在四单元拟合数据上计算，用于检验分裂结构。重新编译后的 80 个 NRMSE 与归档值完全一致。

为保持与归档实验一致，独立高斯噪声的标准差设置为 `1e-4 * std(y)`，实际抽样 RMS 随种子略有变化；奇对称噪声与其实际 RMS 匹配。全部单树参数见 `reproduce.py`。

新增连续数据用五个均匀分布在 `[-1, 1]` 的几何变量生成响应：

```text
y = beta * (2 + 1.25*sin(pi*z0) + 0.7*z1**2 - 1.1*z2*z3 + 0.45*cos(2*z4))
```

每个几何样本取 `beta = -1, 0, +1`。训练集和验证集分别使用种子 20261009、20261010，各有 256 个独立几何样本、768 条记录。噪声种子为 11，实际 RMS 为 `1e-4 * std(y_train)`；模型训练 300 轮。验证误差按奇对称投影后的预测计算，并以未扰动验证标签的标准差归一化。

| 程序库与标签 | 几何分裂数 | 验证 NRMSE |
| --- | ---: | ---: |
| 原始版，未扰动 | 0 | 0.396412 |
| 原始版，独立扰动 | 8160 | 0.107155 |
| 修补版，未扰动 | 8153 | 0.108065 |
| 修补版，独立扰动 | 8160 | 0.107155 |

连续数据检验模型在新合成几何样本上的表现，不代表 CFD 结果，也不代替论文的气动测试集。

## 输出与判断

`build/build_manifest.json` 保存本次编译命令、工具版本、源码及动态库校验值。`build/commands.log` 记录全部构建子进程，各版本另有 `configure.log` 和 `compile.log`。成功及失败命令均保留标准输出和标准错误；超时保留已捕获的输出，异常信息注明日志路径。二进制 Git 归档内容不写入诊断日志，其标准错误仍会保留。

`runs/full/` 保存函数生成的 CSV 输入、两种程序库的运行日志、结果表、协议和 `verification.json`。完整验证成功时退出码为零，报告 `status: PASS`、80 次单树拟合、4 次连续数据拟合，以及 17 项检查通过。smoke 检查 16 次单树拟合和相同的四种连续数据组合。失败时应先查看断言和日志，脚本不会自动重试。

`expected/rebuilt_*` 保留 v1.0.0 原运行记录及当时的源码校验值，其中原验证报告包含 9 项检查。`expected/revalidated_v1.0.1.json` 和 `expected/revalidated_v1.0.2.json` 分别保存后续各版检查及对应验证器的校验值，旧记录保持不变。

原补丁、原消融动态库和本次重编译动态库的校验值分别记录。编译目录或时间改变可能使动态库文件校验值改变；复现判断依据数值和分裂行为，不要求新二进制与原文件逐字节一致。

上游固定为 `v4.6.0`，提交 `d02a01ac6f51d36c9e62388243bcb75c3b1b1774`。补丁只修改 `src/treelearner/serial_tree_learner.cpp` 中 `SerialTreeLearner::FindBestSplits` 的父节点特征排除继承判断，保留增益计算和分裂约束。它是论文用于检验机理的源码干预，不是 LightGBM 官方发布版本。

两种程序库均采用 Release CPU 编译、Ninja 和 OpenMP；GPU、CUDA、MPI、调试及 sanitizer 关闭，训练使用单线程。脱敏后的原 CMake 缓存用于追溯配置，重新编译应使用 `config/build.json`，不应直接复制旧缓存。

复现脚本及合成数据使用 `LICENSE` 中的 MIT 许可证；LightGBM 的 MIT 许可证保留在 `licenses/LightGBM-MIT.txt`。上游源码及子模块分别保留各自许可证。
