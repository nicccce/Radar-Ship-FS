# 08B 执行 Prompt：改进轨迹回放辅助 PPO

日期：2026-09-22。任务：在 `/root/feature-select/Radar-Ship-FS` 直接实现、训练、评价 `improvement-replay-ppo-v1`，目标是提高特征选择后的识别准确率。用户要求探索优先、允许有记录地尝试，具体阳性原因可以以后分析；不要退回新一轮规划或完整审计。

## 1. 这次为什么这样试

08A实际完成12个基础case和4个horizon16 case。均值BAcc：All-SVC92.6477%、Single-PPO92.6271%、Pair-PPO92.5386%、G1 92.3107%、Random-Pair92.2303%、MI32 92.1645%、horizon16 Pair92.1622%。Pair有相对G1 +1.0454 pp的局部阳性，但Single均值更好，延长Pair路径没有改善。

本轮读日志发现最后更新记录的动作熵仍很接近均匀分布；参数确实更新，不能因此说完全没学习。旧Single每case有150–280条“未来刷新最好J并达到MI水平”的候选回放transition，其中包含43–99条即时降分动作。因此先尝试更直接地学好路径。

**主假设：在Single-PPO上加入改进轨迹的加权动作回放，能提高开发识别准确率，争取超过All-SVC。** 这是SIL启发的任务适配，不声称自模仿本身是新发明。

先读：

- `/root/feature-select/Radar-Ship-FS/documents/research-plan/08b-status-and-next-plan.md`
- `/root/feature-select/Radar-Ship-FS/documents/research-plan/08a-block-rewrite-results.md`
- `/root/feature-select/Radar-Ship-FS/configs/v16n/block_rewrite_ppo_v1.toml`
- `/root/feature-select/Radar-Ship-FS/src/run_block_rewrite.py`
- `/root/feature-select/Radar-Ship-FS/src/radar_ship_fs/ppo/block_rewrite.py`

检查AGENTS.md/git status，保留已有修改。不用重读全部历史研究报告，不启动Pair/H16/效率/显式F其他路线。

## 2. 沿用什么

仅使用source-train：`/root/feature-select/dataset/sim_ship_cr_v16n_2x_noise.train.svm`，hash=`2191fdc297aa49ce6a700df956ec44f68dc13e4c737937b31167543a203d22bb`。3897行、75原始列、训练区清洗后65列，坐标双向映射沿用08A。

复用 `/root/feature-select/Radar-Ship-FS/experiments/block_rewrite_ppo_v1/contexts/outer-fold-{0,1}.json` 的开发划分与评分定义；inner seed仍`2026092201+fold`，3-fold mean BAcc。外层、内层行号和清洗/scaler范围与08A一致。历史source-test不打开。已看过的开发数据允许帮助提出修订，不冒充新test。

模型固定StandardScaler+RBF-SVC：C=1、gamma=1/65、class_weight=balanced、probability=False、tol=1e-3、cache_size=256、max_iter=-1、shrinking=True。图、MI、DT依赖通道及图seed沿用08A；不要为本实验顺手换模型/图特征/K。

动作统一Single：从动作前S删一项，从原补集加一项，K32；H8，合法动作即执行，下降也不拒绝。best-so-far奖励、MI/随机/精英重启完全沿用08A：前16 episodes交替MI/随机，之后每4条MI/随机/elite/elite，top8 archive。冻结阶段只用MI/随机起点。

128训练episodes+32冻结episodes，每8条更新，1440次J请求/case；主endpoint是本case所有已评分子集的最高inner J，另报冻结32条轨迹内最好子集，不用outer挑checkpoint或子集。

PPO配置保持hidden64、lr3e-4、gamma=1、GAE=1、clip0.2、epochs4、minibatch64、entropy0.01、value_coef0.5、grad_clip0.5、target_kl0.05，关闭旧feature-ID/prior/proxy奖励。

## 3. 核心新增：从完成的训练轨迹构造回放

一条轨迹为`S_0,a_0,S_1,...,a_(H−1),S_H`，J全部已经算过。对动作a_t的原状态：

```text
b_t = max(J_0,...,J_t)
u_t = max(J_(t+1),...,J_H)
g_t = u_t - b_t
```

保留`g_t>1e-12`且`u_t>=J(MI32)`的transition。保留原selected mask、t/H、J_t、b_t、有序动作(r,a)、g_t和u_t。即时动作可以降分，只要这条实际路径后续刷新最好J；不能悄悄改成仅复制即时涨分动作。

回放仅来自当前case已经完成的**训练**轨迹；不收冻结rollout、不跨fold/seed/臂共享、不加载08A历史轨迹或checkpoint预训练。数据已经存在于当前采集批次，不新增分类器请求。

最多保存512条去重记录。键为完整原状态(selected、progress、current J、best J)+ordered action；重复键保留u最高、再g最高者。容量溢出按u降序、g降序、确定性键排序截断。保存去重前/后计数和被截断数。

每个采集批次完成后：

1. 用刚采集的on-policy buffer执行原PPO更新（适用臂）。
2. 纳入本批已结束轨迹的合格回放样本。
3. 均匀有放回抽batch64，做16个回放梯度步；空池跳过。权重`w=clip(100*g,0,2)`，停止梯度。
4. `L_replay=-sum(w*joint_log_prob(a|s))/sum(w)`，用原Adam优化器lr3e-4，梯度裁剪0.5。回放不计算critic loss、不附加entropy正则。
5. 丢弃本批on-policy buffer，下一批从更新后的策略重新采集；不能用回放后的策略反复套用旧PPO ratio。

使用`ConditionalMacroActorCritic.evaluate_actions`重新计算原有序动作的联合log-prob。回放为独立监督损失，不用旧action的importance ratio，不把off-policy回放塞进`MacroPPOAgent.update`冒充新轨迹。actor及共享encoder更新，critic head无该项梯度。

## 4. 三臂、小矩阵

- `plain_single_ppo`：原Single-PPO，无回放更新。
- `replay_ppo`：原PPO+上述回放，唯一主候选。
- `replay_only`：相同网络/动作/重启，仅上述回放更新，不做PPO；critic输出不参与学习目标。用于识别低成本的“回放就够了”现象，不要求本轮完整归因。

folds=`[0,1]`；新seeds=`[2026092501,2026092502,2026092503]`。18个case全部从头训练，同一fold/seed三臂用同一初始权重：显式设置`torch_seed=seed+100*fold+23`，不要因method label不同调用原METHOD_CODES得到不同初始化。随机起点沿用旧生成规则，三个臂相同；动作/elite/回放采样各自记录RNG。回放不要求消耗相同随机数流。

按seed逐个跑fold0/1，每个fold/seed成组三臂；完成case即可看outer开发结果。新Plain必须重跑，不拿旧seed结果替代主对照。

读取08A相同context/model下的G1/MI32/All-SVC/All-LR指标作为静态参照即可，记录来源，无需重复5284次基线请求。缓存可共享确定性分数，子集archive/轨迹/回放不可跨臂共享。保留请求和真实拟合计数，不开展复杂成本归因研究。

## 5. 预算和允许的一次修订

基础18×1440=25920次J请求，3-fold最多77760次SVC fits。8个CPU评分worker、内层各1线程，GPU可用于小策略网络；每8条episode批量评分，沿用08A已跑通的调度。

先做≤100 fits的短smoke：2个Single episodes、真实PPO+回放更新、保存/加载checkpoint。若当前短轨迹没有合格回放，用纯函数小样本验证梯度，再继续正常采集，不通过降低数据真实性要求伪造阳性样本。

总上限：**110000次classifier fits、计算4小时**；目标一个工作日实现并交付。08A记录约78113 fits/34.7分钟搜索基线墙钟，本轮预估1–3小时计算并给回放训练留余量，不把历史数值当保证。

基础矩阵后若回放池已有足够不同样本（例如≥64）但动作仍接近均匀、Replay-PPO无明显改善，可自主只追加`replay_steps=64`一项修订，同样6个case从头训练，其他不变。最多再25920 fits，基础+修订最多103680，余量留给smoke/端点/修复。一次修订即可，不扫描lr、温度、K或模型。

若策略已经明显改变但outer不升，优先交付该现象，别为压低entropy继续加步数。若资源不足保留完整成组结果和未完成列表；不把partial直接写成所有方向失败。

## 6. 复用接口、新增入口

现有可复用：`run_block_rewrite.py`的`load_train`、`prepare_fold`、`SVCScorer`、`_collect_episode_batch`、archive/评价函数；`block_rewrite.py`的`ConditionalMacroActorCritic(mode="single")`、`MacroPPOAgent`及buffer。旧driver只认识pair_ppo/single_ppo/random_pair，报告也有固定Pair逻辑，不能只改method名字就调用。

新增独立driver并给收集器/训练循环增加明确mode/update hook，或做小范围适配；保留旧默认行为，不复制整个1900行driver。建议文件：

- `/root/feature-select/Radar-Ship-FS/src/run_improvement_replay.py`
- `/root/feature-select/Radar-Ship-FS/src/radar_ship_fs/ppo/improvement_replay.py`
- `/root/feature-select/Radar-Ship-FS/configs/v16n/improvement_replay_ppo_v1.toml`
- `/root/feature-select/Radar-Ship-FS/tests/test_improvement_replay.py`
- `/root/feature-select/Radar-Ship-FS/documents/research-plan/08b-improvement-replay-results.md`

新输出：`/root/feature-select/Radar-Ship-FS/experiments/improvement_replay_ppo_v1/`；smoke另设同级`improvement_replay_ppo_smoke_v1/`。不可覆盖08A报告/配置/checkpoint或用户已有改动。

以下为待实现的新CLI契约，实现后核对--help再执行，不是当前已有参数：

```bash
cd /root/feature-select/Radar-Ship-FS
export PYTHONPATH=src
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
/root/miniconda/envs/dl-lab/bin/python src/run_improvement_replay.py --help
/root/miniconda/envs/dl-lab/bin/python src/run_improvement_replay.py --config configs/v16n/improvement_replay_ppo_v1.toml --stage smoke --jobs 8
/root/miniconda/envs/dl-lab/bin/python src/run_improvement_replay.py --config configs/v16n/improvement_replay_ppo_v1.toml --stage run --jobs 8
# 基于曲线选择有理由的一次修订，配置后才执行：
/root/miniconda/envs/dl-lab/bin/python src/run_improvement_replay.py --config configs/v16n/improvement_replay_ppo_v1.toml --stage refine --jobs 8
/root/miniconda/envs/dl-lab/bin/python src/run_improvement_replay.py --config configs/v16n/improvement_replay_ppo_v1.toml --stage report --jobs 8
```

## 7. 最少检查与结果

定向测试：future-best索引/best-before正确、即时降分但未来刷新记录的动作被保留、无未来改善者不入池、去重与权重正常；回放用旧动作监督而非PPO ratio；单步真实回放提高被监督动作概率；checkpoint恢复包含回放池与RNG。选一个endpoint重新核对分数和坐标即可。

运行新测试、`tests/test_block_rewrite.py`，改到共享评分/映射再补对应已有测试；对新改Python运行ruff。报告实际测试，不复述历史15 passed冒充新验证，不做全仓审计。

保存主与冻结endpoint、所有18个case结果、配对差值、checkpoint、曲线、PPO/回放更新次数、回放池数量与分布、回放NLL、动作熵、成本和配置。补一个无需新分类器拟合的小probe：每case首批固定16个状态/原动作，记录训练前后动作概率及entropy，作为学习行为读数，不设为准确率准入门槛。

主比较`Replay-PPO−Plain-Single-PPO`；fold内平均3seeds，再看两fold均值，原始case全部列出。同时列相对Replay-only、All-SVC、G1和MI32的差值。普通accuracy/各类recall辅助报告。

+0.30～0.50 pp是值得追的量级；均值正向或明显局部阳性都可继续一次，不要求显著性、全五折或完整机制证明。entropy/NLL变好不能替代outer BAcc。Replay-only胜出时明确记录其价值，不能为RL身份改写结果。

阳性后先扩大有希望版本；本版及一次修订均无准确率信号，则后续唯一备选是**保持同一SVC、允许16–48维的add/delete/STOP策略**，检验固定K是否限制识别。本任务不顺手启动该第二路线。

最终报告：准确率是否提高、哪一版本最有希望、策略是否更会利用改进动作、下一处最值得改什么。具体阳性机制可以留待后续。本任务以真实训练结果和可继续的代码/产物结束，不返回一份新的待规划清单。
