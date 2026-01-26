# 大模型辅助科研论文写作质量检查清单

> **使用说明**：本清单用于指导大模型根据实验结果生成科研论文时的质量控制。可将此文件放入项目根目录或`.cursor/rules/`中，让AI在写作时自动参考。
>
> **版本**: v2.0 | **更新日期**: 2026年1月 | **基于实际论文修改经验总结**

---

## 一、数据与实验一致性验证

- [ ] 1.1 核查全文数据与代码运行结果是否一致，标记所有不一致之处
- [ ] 1.2 检查是否存在数据捏造成分，对虚假数据进行全文修正
- [ ] 1.3 修正涉及错误数据计算得出的衍生数据
- [ ] 1.4 同步更新使用相关数据制作的图表
- [ ] 1.5 数据修改后，检查相关语句的逻辑表达和结论是否需要同步修改
- [ ] 1.6 检查全文中同一数据的小数位数是否统一（如R²值统一保留3位小数：0.690而非0.69）
- [ ] 1.7 核查所有声称显著/不显著的结论是否都报告了相应统计量（p值、置信区间、效应量等）
- [ ] 1.8 Methods中描述的样本量与Results中实际报告的是否一致
- [ ] 1.9 代码中的超参数（learning rate、batch size、epoch等）是否与文中描述完全一致
- [ ] 1.10 是否说明了随机种子设置，多次运行结果是否报告了均值和标准差
- [ ] 1.11 Methods中的公式变量是否完整无遗漏，步骤是否有捏造的不存在部分
- [ ] 1.12 **新增**：表格中的数值是否与最新的实验结果CSV文件一致
- [ ] 1.13 **新增**：Abstract中报告的关键数值是否与正文Results一致

---

## 二、图表规范性检查

- [ ] 2.1 Results部分的结果与图表是否与代码输出一致
- [ ] 2.2 所有图表在正文中是否都有文字提及和解释
- [ ] 2.3 若对某图片进行了修改，追查是否对其他相关图也进行了修改
- [ ] 2.4 图的caption是否能独立理解，不需要回到正文查看
- [ ] 2.5 所有图的坐标轴是否都有标签和单位
- [ ] 2.6 正文中Figure/Table的引用顺序是否与出现顺序一致（Fig.1应在Fig.2之前被引用）
- [ ] 2.7 图表在黑白打印时是否仍可区分不同数据系列
- [ ] 2.8 图片分辨率是否符合目标期刊要求（通常300dpi以上）
- [ ] 2.9 表格数据对齐方式是否一致（数字右对齐，文字左对齐）
- [ ] 2.10 图表编号是否连续，无跳号或重复
- [ ] 2.11 **新增**：图表路径是否正确指向实际文件位置
- [ ] 2.12 **新增**：Appendix中的图表是否使用正确的编号格式（如Figure A1, Table A1）
- [ ] 2.13 **新增**：图表caption是否使用加粗的标题格式（如 **\textbf{Title.}** Description）
- [ ] 2.14 **新增**：多子图的figure是否正确标注(A), (B), (C)等

---

## 三、文献引用规范

- [ ] 3.1 Introduction部分的逻辑重构后，在适当位置给出真实引用
- [ ] 3.2 检查所采纳的引用是否与其所在句子的内容相关
- [ ] 3.3 "有研究表明"类句子后是否有参考文献支撑
- [ ] 3.4 "有很多研究..."类表达是否有多个参考文献（而非仅一个）
- [ ] 3.5 检查近3-5年文献占比是否足够，是否遗漏领域内重要新进展
- [ ] 3.6 自引比例是否合理（通常不超过10-15%）
- [ ] 3.7 全文引用格式是否统一（如et al.写法、作者名格式等）
- [ ] 3.8 引用是否放在句末正确位置，多个引用的排列顺序是否符合期刊要求
- [ ] 3.9 重要论断是否有多个独立来源支持（避免孤证）
- [ ] 3.10 上网搜索最近几年的相关研究及领域内有影响力的研究作为补充引用
- [ ] 3.11 引用的文献是否确实存在且内容与引用目的相符（防止AI幻觉）
- [ ] 3.12 **新增**：删除.bib文件中未被引用的无关文献条目
- [ ] 3.13 **新增**：核心方法是否引用了原始论文（如Adam优化器引用Kingma2014，Attention引用Vaswani2017）
- [ ] 3.14 **新增**：Related Work中引用的文献是否覆盖了主要研究方向

---

## 四、学术语言规范

- [ ] 4.1 删除AI痕迹：去掉分点常用的破折号"-"
- [ ] 4.2 删除主观性/情绪性词语：显著的、重大的、有意义的（significant除外）、令人兴奋的等
- [ ] 4.3 检查并删除口语化/AI化词语：here、indeed、notably、interestingly等
- [ ] 4.4 确保文章使用科学专业的语言表达
- [ ] 4.5 检查时态一致性：Methods用过去时，普遍事实用现在时
- [ ] 4.6 检查人称一致性：全文使用"we"还是被动语态是否统一
- [ ] 4.7 检查术语一致性：同一概念全文是否使用同一术语（如"Explainable Model"不要与"explainable model"混用）
- [ ] 4.8 避免绝对化表述：删除"prove"（改用demonstrate/suggest）、"always"、"never"等
- [ ] 4.9 量化表达：将"很多/一些/少量"替换为具体数字或比例
- [ ] 4.10 因果表达谨慎：若仅有相关性证据，使用"is associated with"而非"cause/lead to"
- [ ] 4.11 删除空洞强调词："It is worth noting that"、"Importantly"（除非确实重要）
- [ ] 4.12 避免过度概括的开头：如"In recent years, X has attracted significant attention"
- [ ] 4.13 检查逻辑连接词是否滥用：Furthermore、Moreover、Additionally等
- [ ] 4.14 检查hedging词使用是否恰当：may、might、could的使用频率和位置
- [ ] 4.15 检查句式是否单一：避免大量使用"This paper proposes..."等相似结构
- [ ] 4.16 **新增**：使用grep命令批量检查AI痕迹词语（见附录命令）

---

## 五、结构与逻辑检查

- [ ] 5.1 检查文章是否构成一个整体：研究问题、实验方法、实验结果、结论是否有明显逻辑关系
- [ ] 5.2 Introduction提出的每个研究问题，在Results和Discussion中是否都有对应回答
- [ ] 5.3 本文贡献点在Introduction末尾和Conclusion中是否清晰列出且一致
- [ ] 5.4 Limitation部分是否真实反映研究不足，而非敷衍
- [ ] 5.5 每段首句是否能概括该段主旨（主题句）
- [ ] 5.6 段落之间、章节之间是否有逻辑连接词或过渡句
- [ ] 5.7 Discussion部分逻辑是否合适，是否需要重构
- [ ] 5.8 Conclusion部分是否与全文发现相符，无过度延伸
- [ ] 5.9 修改后人工阅读全文，检查逻辑流是否通顺
- [ ] 5.10 **新增**：是否包含Related Work section（计算机/ML领域必需）
- [ ] 5.11 **新增**：是否包含独立的Experiments section（区别于Methods）
- [ ] 5.12 **新增**：Introduction末尾是否有明确的贡献点列表（通常4-5点）

---

## 六、格式与投稿准备

- [ ] 6.1 字数/页数是否符合目标期刊要求
- [ ] 6.2 摘要是否包含背景、目的、方法、结果、结论五要素，且可独立理解
- [ ] 6.3 关键词是否涵盖主要概念，是否与领域通用术语一致
- [ ] 6.4 所有缩写首次出现时是否给出全称（如MLP、RBF、KNN、SVR等）
- [ ] 6.5 致谢与funding声明是否包含必要的基金资助信息
- [ ] 6.6 作者贡献声明是否符合期刊的CRediT要求
- [ ] 6.7 利益冲突声明是否按期刊要求添加
- [ ] 6.8 参考文献格式是否符合目标期刊要求
- [ ] 6.9 章节标题格式、字体、字号是否符合期刊模板
- [ ] 6.10 **新增**：是否包含Data Availability Statement
- [ ] 6.11 **新增**：是否包含Author Contributions声明
- [ ] 6.12 **新增**：是否包含Conflict of Interest声明
- [ ] 6.13 **新增**：是否包含Acknowledgments部分
- [ ] 6.14 **新增**：作者信息是否完整（姓名、单位、邮箱）
- [ ] 6.15 **新增**：LaTeX编译是否无错误（仅允许minor warnings）

---

## 七、学术诚信检查

- [ ] 7.1 使用Turnitin等工具检查相似度
- [ ] 7.2 确认所有图表均为原创或已获得授权并正确引用
- [ ] 7.3 是否说明数据和代码的开放获取方式（如适用）
- [ ] 7.4 确认无抄袭、自我抄袭或一稿多投问题
- [ ] 7.5 确认所有作者均对论文有实质贡献且同意投稿

---

## 八、最终审读

- [ ] 8.1 全文通读，检查语法和拼写错误
- [ ] 8.2 检查标点符号使用是否正确（特别是中英文标点混用问题）
- [ ] 8.3 检查数学公式编号是否连续，公式引用是否正确
- [ ] 8.4 检查脚注/尾注（如有）格式是否统一
- [ ] 8.5 最终确认：文件格式、命名是否符合投稿系统要求
- [ ] 8.6 **新增**：PDF编译后检查所有交叉引用是否正确解析（无"??"）
- [ ] 8.7 **新增**：检查LaTeX特殊字符是否正确转义（如%、&、_等）

---

## 九、计算机/机器学习领域特定检查（新增）

### 9.1 论文结构完整性

标准ML论文结构应包含：

```
1. Abstract
2. Introduction (含贡献点列表)
3. Related Work ← 必需
   - 3.1 [研究方向1相关工作]
   - 3.2 [研究方向2相关工作]
   - 3.3 [本文方法相关工作]
4. Methods / Methodology
5. Experiments ← 建议独立成节
   - 5.1 Experimental Setup
   - 5.2 Datasets
   - 5.3 Baselines
   - 5.4 Evaluation Metrics
   - 5.5 Implementation Details
6. Results
7. Discussion / Analysis
8. Conclusion
9. References
10. Appendix (可选)
```

### 9.2 实验设置检查

- [ ] 9.2.1 是否报告了计算平台（CPU/GPU型号、内存）
- [ ] 9.2.2 是否报告了软件环境（Python版本、框架版本）
- [ ] 9.2.3 是否报告了随机种子
- [ ] 9.2.4 是否报告了训练时间或计算复杂度
- [ ] 9.2.5 是否提供了超参数搜索范围或选择依据
- [ ] 9.2.6 是否说明了数据集划分方式（train/val/test比例）

### 9.3 模型比较公平性

- [ ] 9.3.1 所有baseline是否使用相同的数据预处理
- [ ] 9.3.2 所有模型是否使用相同的评估指标
- [ ] 9.3.3 是否报告了baseline的来源（原论文/自己实现）
- [ ] 9.3.4 深度学习模型是否报告了多次运行的均值±标准差

### 9.4 可复现性

- [ ] 9.4.1 是否提供代码开源链接（或承诺发布）
- [ ] 9.4.2 是否提供数据集获取方式
- [ ] 9.4.3 关键实现细节是否足够详细以供复现

---

## 十、Related Work写作指南（新增）

### 10.1 Related Work的作用

1. **定位研究贡献**：说明本文与现有工作的区别
2. **展示领域知识**：证明作者对领域的了解
3. **建立研究动机**：说明为什么需要本文的方法

### 10.2 Related Work的组织方式

**按主题组织**（推荐）：

```
2.1 [主题A]相关工作
2.2 [主题B]相关工作
2.3 [主题C]相关工作
```

**按时间组织**：

```
2.1 早期方法
2.2 近期进展
2.3 最新研究
```

### 10.3 Related Work写作要点

- [ ] 每个子节末尾说明与本文的关系/区别
- [ ] 避免简单罗列文献，要有分析和比较
- [ ] 引用数量适中（通常20-40篇）
- [ ] 覆盖经典工作和最新进展
- [ ] 对每类方法的优缺点有客观评价

### 10.4 Related Work常见子节（ML领域）

根据研究内容选择相关子节：

| 研究方向 | 可能的子节 |
|---------|-----------|
| 深度学习 | Neural Network Architectures, Optimization Methods |
| 可解释AI | Explainable AI, Attention Mechanisms, Feature Importance |
| 领域应用 | [Domain]-specific Machine Learning |
| 物理信息 | Physics-Informed Machine Learning |
| 集成学习 | Ensemble Methods, Random Forests |
| 时序预测 | Time Series Forecasting, Recurrent Networks |

---

## 附录A：常见AI写作痕迹速查表

| 问题类型 | 常见示例 |
|---------|---------|
| 空洞强调词 | It is worth noting that, Importantly, Notably, Interestingly, Remarkably |
| 过度概括 | In recent years..., has attracted significant attention, has been widely studied |
| 连接词滥用 | Furthermore, Moreover, Additionally, In addition, Besides (连续使用) |
| 情绪性词语 | exciting, remarkable, groundbreaking, revolutionary, unprecedented |
| 绝对化表述 | prove, always, never, undoubtedly, certainly, definitely |
| 口语化词语 | here, indeed, actually, basically, really, just, quite |
| 冗余表达 | in order to (→ to), due to the fact that (→ because), at this point in time (→ now) |
| 被动滥用 | It was found that... It is believed that... It can be seen that... |

---

## 附录B：学术写作替换词表

| 避免使用 | 推荐替换 |
|---------|---------|
| prove | demonstrate, show, suggest, indicate |
| very/really + adj | 使用更精确的形容词 |
| a lot of / lots of | numerous, substantial, considerable |
| thing | factor, aspect, element, component |
| get | obtain, acquire, achieve |
| big/small | substantial/minimal, significant/negligible |
| good/bad | effective/ineffective, beneficial/detrimental |
| show | demonstrate, illustrate, reveal, indicate |

---

## 附录C：AI痕迹检测命令（新增）

在终端中运行以下命令检测AI写作痕迹：

```bash
# 检测空洞强调词
grep -i -E "notably|interestingly|importantly|remarkably|it is worth noting" paper.tex

# 检测过度概括
grep -i -E "in recent years|has attracted significant attention|has been widely studied" paper.tex

# 检测绝对化表述
grep -i -E "\bprove\b|\balways\b|\bnever\b|\bundoubtedly\b|\bcertainly\b" paper.tex

# 检测口语化词语
grep -i -E "\bhere\b|\bindeed\b|\bactually\b|\bbasically\b" paper.tex

# 检测连接词滥用（需人工判断是否过度）
grep -i -E "furthermore|moreover|additionally" paper.tex | wc -l
```

---

## 附录D：必需引用的经典论文（ML领域，新增）

根据使用的方法，确保引用以下经典论文：

| 方法/技术 | 必需引用 |
|----------|---------|
| Adam优化器 | Kingma & Ba, 2014 |
| Attention机制 | Vaswani et al., 2017 (Transformer) |
| Random Forest | Breiman, 2001 |
| Dropout | Srivastava et al., 2014 |
| Batch Normalization | Ioffe & Szegedy, 2015 |
| ResNet | He et al., 2016 |
| LSTM | Hochreiter & Schmidhuber, 1997 |
| GAN | Goodfellow et al., 2014 |
| BERT | Devlin et al., 2019 |
| Physics-Informed NN | Raissi et al., 2019 |
| SHAP | Lundberg & Lee, 2017 |
| XGBoost | Chen & Guestrin, 2016 |
| SVR/SVM | Cortes & Vapnik, 1995 |

---

## 附录E：LaTeX论文必需声明模板（新增）

```latex
\section*{Data Availability Statement}
The dataset and source code used in this study are available from 
the corresponding author upon reasonable request. [或提供GitHub链接]

\section*{Author Contributions}
\textbf{Author1}: Conceptualization, Methodology, Software, 
Formal analysis, Writing - Original Draft. 
\textbf{Author2}: Data curation, Validation, Visualization. 
\textbf{Author3}: Supervision, Writing - Review \& Editing, 
Project administration.

\section*{Conflict of Interest}
The authors declare that they have no known competing financial 
interests or personal relationships that could have appeared to 
influence the work reported in this paper.

\section*{Acknowledgments}
The authors acknowledge the support from [Funding Agency] under 
Grant No. [Grant Number]. We thank [Person/Organization] for 
[specific contribution].
```

---

## 附录F：论文修改检查流程（新增）

建议按以下顺序进行论文检查：

```
1. 结构完整性检查
   ├── 是否包含所有必需章节
   ├── 是否有Related Work
   └── 是否有贡献点列表

2. 数据一致性检查
   ├── 表格数据与CSV文件对比
   ├── Abstract数据与正文对比
   └── 小数位数统一

3. 引用规范检查
   ├── 删除无关引用
   ├── 添加方法原始引用
   └── 检查引用顺序

4. 语言规范检查
   ├── 时态一致性（Methods用过去时）
   ├── 术语一致性
   └── AI痕迹检测

5. 格式完整性检查
   ├── 缩写定义
   ├── 必需声明（Data, Author, Conflict, Acknowledgments）
   └── 图表编号和引用

6. 编译检查
   ├── LaTeX编译无错误
   ├── 交叉引用正确
   └── PDF输出正常
```

---

## 更新日志

| 版本 | 日期 | 更新内容 |
|-----|------|---------|
| v2.0 | 2026-01 | 基于实际论文修改经验大幅扩展：新增Related Work指南、ML领域特定检查、必需引用列表、LaTeX模板、检查流程等 |
| v1.0 | 2025-01 | 初始版本 |

---

*本清单基于实际论文修改经验总结，持续更新中。*
