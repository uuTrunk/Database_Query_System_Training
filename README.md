# Database Query System - Training (并发预测调度与离线模型中枢)

此独立项目包含了由 Controller 中完整剥离出的**并发调度机器学习微服务**。目前采用一套精简的 **Django API** 支持了前端系统对于自然语言问诊的单次生成胜率判断，以及提供了纯净的、去耦合的**离线样本复用及再培训平台**。

## 核心架构构成

本项目由两大部分组成：

### 1. 并发预测模型调度微服务 (Django Backend `8001` 端口)
- **`training_backend/`**: Django 项目的设定目录。
- **`api/`**: 提供一个 `http://localhost:8001/api/predict` 预测接口。
- **作用**: 在接到前端 `Controller` 的问句以后，后端代码会利用基于先前预处理的 BERT（多语言版）回归加载本地的 `saves/model_best.pth` 进行预测分析。最终按照推算，返还能使得系统整体生成成功率达 `99%` 的推荐线程利用数（限制由于接口瓶颈，会在 1~5 的范围内）。

### 2. 数据生产流水线 (Training Pipelines)
- **`training/`**: 存放了模型机器学习流水线相关的纯 Python 脚本，包括数据采样、`gen_training_questions.py`标签抽离与最终执行深度下落预测权重的生成(`model.py`)。 
- **作用**: 负责利用离线日志重取特征值进行自动再回炉分析，它在执行过程中不再占用服务网络。将得到的最佳新版 BERT 隐层结构替换当前文件夹内 `saves/` 里的权重。

## 环境配置与运行

该微服务及全部脱机模型操作脚本需要建立在原架构下的 Conda 环境 `dqs`，其已经满足了 `PyTorch`, `Transformers`, `Django` 的库包标准。

### 启动预测服务器

打开终端，激活环境并启动入口（`manage.py` 已全局正名为极简版 `main.py`）：
```bash
conda activate dqs
python main.py runserver 8001
```

### 完整生成、更新日志或迭代模型权重的脱机操作

由于架构中所有系统不再共享目录或底层加载包，离线调用数据取样仍需使用 `Database_Query_System_Agent` 中的模块。

如果需要在当前位置获取最新的模型产出，请首先确认你的 Agent 端口 `8000` 仍处于存活与监听状态，而后你可以依据需要按序启动离线 Python 程序：
```bash
conda activate dqs
# 步骤 1：生成或更新训练问题集缓存
python -m training.gen_training_questions 
# 步骤 2：生成或更新基于 Agent 的脱机成功率验证大模型分析指标底库（最持久的环节）
python -m training.test_ask_graph 
# 步骤 3：正式启动并取代现有的 .pth 存档
python -m training.model
```
重新训练完毕后将覆写 `saves/` 目录下存放的最佳回溯值。完成后需重新热更新启动您的 8001 `main.py runserver` 来搭载新鲜权重的推流预判。
