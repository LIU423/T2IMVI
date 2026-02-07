# T2IMVI - Text-to-Image Metaphor Visual Interpretation

T2IMVI 是一个用于分析和量化文本到图像模型对隐喻/习语理解能力的完整流水线框架。

## 项目结构

```
T2IMVI/
├── project_config.py           # 集中路径配置（核心配置文件）
├── README.md                   # 本文件
│
├── data/                       # 数据目录
│   ├── input/                  # 输入数据
│   │   └── IRFL/              # IRFL数据集
│   │       ├── non_none/      # 非空数据
│   │       └── matched_images/ # 匹配的图像
│   ├── output/                 # 输出数据
│   │   ├── IRFL/              # IRFL处理结果
│   │   ├── phase0/            # Phase 0 输出
│   │   └── phase1_extraction/ # Phase 1 提取输出
│   └── prompt/                 # 提示词模板
│
├── quantification_pipeline/    # 量化流水线（核心）
│   ├── phase0/                # Phase 0: 预处理
│   │   ├── imageability/      # 可成像性分析
│   │   ├── transparency/      # 透明度分析
│   │   └── run_phase0.py      # Phase 0 主入口
│   ├── phase1_extraction/     # Phase 1: 视觉元素提取
│   ├── phase1_scoring/        # Phase 1: 评分
│   ├── phase2_calculation/    # Phase 2: 指标计算
│   └── score_total.py         # 综合评分
│
├── reliability_analysis/       # 可靠性分析实验
│   ├── main.py                # 实验主入口
│   ├── config.py              # 实验配置
│   ├── experiment1_ranking.py # 实验1: 排名对齐
│   ├── experiment2_stability.py # 实验2: 分数稳定性
│   ├── metrics.py             # 评估指标
│   └── results/               # 实验结果
│
└── example/                    # 示例文件
```

## 模块功能说明

### 1. 集中配置 (`project_config.py`)

**功能**: 统一管理项目中所有路径配置，实现项目的可移植性。

**核心特性**:
- 自动检测项目根目录
- 所有路径从根目录派生
- 无需修改代码即可迁移项目

**使用方法**:
```python
from project_config import PROJECT_ROOT, DATA_DIR, OUTPUT_DIR

# 使用路径
input_file = DATA_DIR / "input" / "my_data.json"
```

**验证配置**:
```bash
python project_config.py
```

---

### 2. Phase 0: 预处理 (`quantification_pipeline/phase0/`)

**功能**: 对习语进行预分析，评估其可成像性(imageability)和透明度(transparency)。

| 子模块 | 功能 |
|--------|------|
| `imageability/` | 评估习语的视觉可表达程度 |
| `transparency/` | 评估习语字面义与隐喻义的透明程度 |
| `run_phase0.py` | 统一运行入口 |

**运行方式**:
```bash
cd D:\Opencode\T2IMVI
python quantification_pipeline/phase0/run_phase0.py
```

---

### 3. Phase 1 Extraction: 视觉元素提取 (`quantification_pipeline/phase1_extraction/`)

**功能**: 使用LLM从习语中提取视觉元素，生成两条并行轨道：

| 轨道 | 描述 |
|------|------|
| **Literal Track** | 基于字面词汇的形态可视化 |
| **Figurative Track** | 基于隐喻含义的视觉知识图谱 |

**运行方式**:
```bash
# 测试模式（处理1个习语）
python quantification_pipeline/phase1_extraction/main.py --test

# 完整提取
python quantification_pipeline/phase1_extraction/main.py --mode both

# 仅字面轨道
python quantification_pipeline/phase1_extraction/main.py --mode literal

# 仅隐喻轨道
python quantification_pipeline/phase1_extraction/main.py --mode figurative
```

**高级选项**:
```bash
--limit 10          # 限制处理数量
--ids 1 2 3         # 处理指定ID
--reset             # 重置检查点
--model MODEL_PATH  # 使用不同模型
--device cuda:0     # 指定GPU
```

---

### 4. Phase 1 Scoring: 评分 (`quantification_pipeline/phase1_scoring/`)

**功能**: 对提取的视觉元素进行质量评分和验证。

| 组件 | 功能 |
|------|------|
| `evaluator.py` | 评分逻辑 |
| `verifiers/` | 验证器 |
| `models/` | 评分模型 |

**运行方式**:
```bash
python quantification_pipeline/phase1_scoring/main.py
```

---

### 5. Phase 2 Calculation: 指标计算 (`quantification_pipeline/phase2_calculation/`)

**功能**: 计算最终的量化指标。

| 指标 | 描述 |
|------|------|
| VC | Visual Correspondence (视觉对应) |
| CA | Conceptual Alignment (概念对齐) |
| CR | Compositional Reasoning (组合推理) |
| IU | Image Understanding (图像理解) |
| AEA | Aesthetic-Emotional Alignment (美学情感对齐) |

**运行方式**:
```bash
# 运行主评估
python quantification_pipeline/phase2_calculation/main.py

# 计算 VC/CA/CR 分数
python quantification_pipeline/phase2_calculation/score_vc_ca_cr.py

# 计算 IU 分数
python quantification_pipeline/phase2_calculation/iu_main.py
```

---

### 6. 综合评分 (`quantification_pipeline/score_total.py`)

**功能**: 汇总所有阶段的分数，生成最终评估报告。

**运行方式**:
```bash
python quantification_pipeline/score_total.py
```

---

### 7. 可靠性分析 (`reliability_analysis/`)

**功能**: 验证评估方法的可靠性和稳定性。

| 实验 | 描述 |
|------|------|
| Experiment 1 | 排名对齐分析 |
| Experiment 2 | 分数稳定性分析 |

**运行方式**:
```bash
# 验证数据加载
python reliability_analysis/main.py verify

# 运行实验1
python reliability_analysis/main.py exp1 -m qwen3_vl_2b_T2IMVI

# 运行实验2
python reliability_analysis/main.py exp2 --self-test qwen3_vl_2b_T2IMVI

# 运行所有实验
python reliability_analysis/main.py all -m qwen3_vl_2b_T2IMVI

# 显示/导出配置
python reliability_analysis/main.py config
python reliability_analysis/main.py config --export
```

---

## 环境配置

### 1. 创建虚拟环境

```bash
# 使用 conda
conda create -n T2IMVI python=3.11
conda activate T2IMVI

# 或使用 venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 2. 安装依赖

```bash
# 核心依赖
pip install torch torchvision transformers
pip install pydantic tqdm pandas numpy

# 可靠性分析依赖
pip install -r reliability_analysis/requirements.txt
```

### 3. 验证安装

```bash
# 验证路径配置
python project_config.py

# 验证可靠性分析模块
python reliability_analysis/main.py verify
```

---

## 路径配置说明

### 集中配置文件 (`project_config.py`)

所有路径都从 `PROJECT_ROOT` 自动派生，无需手动修改。

| 变量名 | 描述 |
|--------|------|
| `PROJECT_ROOT` | 项目根目录（自动检测） |
| `DATA_DIR` | 数据目录 |
| `INPUT_DIR` | 输入数据目录 |
| `OUTPUT_DIR` | 输出数据目录 |
| `PROMPT_DIR` | 提示词模板目录 |
| `OUTPUT_PHASE0_DIR` | Phase 0 输出目录 |
| `OUTPUT_PHASE1_EXTRACTION_DIR` | Phase 1 提取输出目录 |
| `RELIABILITY_ANALYSIS_DIR` | 可靠性分析目录 |
| `QUANTIFICATION_PIPELINE_DIR` | 量化流水线目录 |

### 在子模块中使用

```python
# 方法1: 直接导入（推荐）
from project_config import PROJECT_ROOT, DATA_DIR, OUTPUT_DIR

# 方法2: 相对路径导入
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).parent.resolve()
_PROJECT_ROOT = _THIS_DIR.parent.parent  # 根据深度调整
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from project_config import *
```

### 添加新路径

在 `project_config.py` 中添加:

```python
# 新路径定义
MY_NEW_DIR = PROJECT_ROOT / "my_new_folder"

# 更新 get_all_paths() 函数
def get_all_paths():
    return {
        ...
        "MY_NEW_DIR": str(MY_NEW_DIR),
    }
```

---

## 项目迁移

本项目支持零修改迁移。将整个 `T2IMVI` 文件夹复制到新位置后：

1. **无需修改任何代码** - `PROJECT_ROOT` 自动检测
2. **激活虚拟环境** - 确保依赖已安装
3. **验证配置** - 运行 `python project_config.py`

```bash
# 迁移后验证
cd /new/path/T2IMVI
python project_config.py
# 应显示新路径
```

---

## 完整工作流程

```bash
# 1. 激活环境
conda activate T2IMVI

# 2. 进入项目目录
cd D:\Opencode\T2IMVI

# 3. Phase 0: 预处理
python quantification_pipeline/phase0/run_phase0.py

# 4. Phase 1: 视觉元素提取
python quantification_pipeline/phase1_extraction/main.py --mode both

# 5. Phase 1: 评分
python quantification_pipeline/phase1_scoring/main.py

# 6. Phase 2: 指标计算
python quantification_pipeline/phase2_calculation/main.py
python quantification_pipeline/phase2_calculation/score_vc_ca_cr.py

# 7. 综合评分
python quantification_pipeline/score_total.py

# 8. 可靠性分析（可选）
python reliability_analysis/main.py all -m qwen3_vl_2b_T2IMVI
```

---

## 常见问题

### Q: 导入错误 `ModuleNotFoundError`

确保从项目根目录运行，或在脚本中正确设置路径:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))
```

### Q: 路径不存在

运行 `project_config.py` 检查配置，使用 `validate_paths(create_missing=True)` 创建缺失目录:

```python
from project_config import validate_paths
validate_paths(create_missing=True)
```

### Q: CUDA 内存不足

减少批处理大小或使用 `--device cpu`:

```bash
python main.py --device cpu
```

---

## 许可证

[待定]

## 作者

T2IMVI Team
