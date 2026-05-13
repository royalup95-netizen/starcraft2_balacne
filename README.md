# StarCraft II Balance Analysis with Bayesian Statistics

## 项目简介
利用贝叶斯统计方法对2020-2021年星际争霸2比赛数据进行分析，验证社区中关于“P IMBA”或“T IMBA”的讨论。

## 研究问题
1. 各对阵（PvT、PvZ、TvZ）是否存在统计显著的失衡？
2. Protoss是否存在轻微优势倾向？
3. 玩家人口结构是否会对平衡性统计产生影响？

## 数据来源与致谢

本研究所使用的比赛数据来源于 **Liquipedia** (https://liquipedia.net/starcraft2)

根据 Liquipedia API 的数据使用条款：
- 所有数据仅用于学术研究目的
- 本仓库不包含从API获取的原始数据
- 分析结果和可视化图表已妥善标注数据来源

感谢 Liquipedia 提供的数据支持。

## 方法
- 动态层次贝叶斯逻辑回归模型
- No-U-Turn Sampler (NUTS)进行MCMC采样
- 94%最高后验密度区间（HDI）进行不确定性量化

## 主要结论
1. StarCraft II整体较为平衡（所有对阵94% HDI均覆盖50%）
2. Protoss存在轻微优势倾向（PvT 52.66%，PvZ 52.76%），但效应量很小
3. 人口效应显著存在（玩家基数大的种族更易获得统计优势）
4. 2020-2021年平衡性时间动态较弱

## 文件结构
starcraft2_balance/
├── code/
│ ├── data_collection.py # API数据采集
│ └── bayesian_model.py # 贝叶斯模型核心代码
├── results/
│ ├── posterior_matchup.png # 后验分布图
│ └── sc2_bayesian_analysis.png
├── report/
│ └── balance_analysis.pdf # 论文PDF
├── README.md
├── LICENSE
└── requirements.txt

#模型缺点以及改进方向
这份报告只是我用来做课程作业，所以模型上有很多漏洞，比如说没有考虑到地图池的问题、没有最新的数据支撑以及各项平衡性补丁的细致研究。本文只是提供一些可参考的思路，如果有任何疑问和批评，本人都欢迎讨论。

## 贡献者

- 钱宁康 - 北京师范大学 - 数据分析与报告撰写

## 项目状态

✅ 已完成

## 如何引用

Qian, N. (2026). StarCraft II Race Balance Analysis: A Bayesian Approach. GitHub repository.
