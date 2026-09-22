# 首轮执行 Prompt：直接尝试成对重组 PPO，先找准确率阳性

日期：2026-09-22。版本：`accuracy-exploration-v3`。适用于没有此前对话上下文的 agent。

你要在 `/root/feature-select/Radar-Ship-FS` **直接实现、训练并评价 `block-rewrite-ppo-v1`**。用户的目标是提高强化学习特征选择后的识别准确率，最新要求是“探索更激进，重点关注创新和尝试，阳性的原因可以晚些分析”。

因此，你应尽快做出能跑的成对重组PPO，与少量关键对照比较，并可根据结果做一次有理由的修订。不要先跑非RL前瞻来决定是否允许训练，不要求完整五折/机制证明/DFS重建完成才开工，不把任务改成效率研究或又一份规划。所有本轮结果按开发探索报告。

## 1. 首要假设与必要背景

**H：一次联合选择两个删除项和两个新增项，并允许连续搜索暂时降分，能比单交换PPO和贪心找到识别准确率更高的32维子集。** 第一版的创新尝试是条件化块动作、可经过低谷的轨迹及精英混合重启。它是待验证的方法原型，不预设RL有效或声称首创。

已知source-train为3897行、75原始列，历史清洗后65列；scene/parent/env缺失。4B的LR BAcc评分器有信息，NO-GO来自联合搜索收益不足。4C只检查LR、K32、forward后最多两轮完整single-swap，平均+0.1009 pp，没有训练RL，更没有排除所有组合路径。历史All-LR=91.8579%、forward+swap=91.6751%。旧分支停止决定保持，新方法独立命名。

上一版SG-8效率方案和“非RL双步前瞻先过门槛”的方案均撤销，未执行。旧source-test反复使用，本轮不用。已有outer folds也属于开发数据，允许据此形成新想法，不能当作最终新测试集。

## 2. 只读足够开工的材料

检查适用AGENTS.md与git status，保留已有修改。完整读：

- `/root/feature-select/Radar-Ship-FS/documents/research-plan/08-exploration-plan.md`
- `/root/feature-select/Radar-Ship-FS/documents/research-plan/04c-k32-robustness.md`

定向查阅以下真实接口，避免重新通读整个研究历史：

- `/root/feature-select/Radar-Ship-FS/src/run_k32_robustness.py`：`load_raw`、`clean_on_outer_train`、`exact_k_forward`、`best_improvement_swaps`、`fit_lr_predict`。
- `/root/feature-select/Radar-Ship-FS/src/run_reward_alignment.py`：`canonical_subset`、`all_single_swaps`。
- `/root/feature-select/Radar-Ship-FS/src/radar_ship_fs/ppo/ppo_model.py`：`GraphActorCritic`、`_encode`及图池化；默认prior必须关闭。
- `/root/feature-select/Radar-Ship-FS/src/radar_ship_fs/ppo/ppo_agent.py`：PPO clip/GAE/update逻辑；**现有buffer只保存标量action，不能直接承载四元宏动作**。
- `/root/feature-select/Radar-Ship-FS/src/radar_ship_fs/ppo/ppo_graph.py`：`build_feature_graph`，已有MI/相关性/DT依赖通道。
- `/root/feature-select/Radar-Ship-FS/src/radar_ship_fs/ppo/ppo_env.py`：仅参考结构；旧代理shaping、候选池与终局accuracy不适用于本实验。
- `/root/feature-select/Radar-Ship-FS/src/radar_ship_fs/feature_mapping.py`。

这些路径已核对。旧`run_k32_robustness.py`的CLI为`--stage {prepare,run,finalize,all}`、`--jobs`；prepare有3A保护，run会执行旧实验，不要运行旧stage或删除保护。也不要使用会同时加载source-train/test的旧通用PPO入口。新driver只加载train。

## 3. 数据与共同评分器

数据：`/root/feature-select/dataset/sim_ship_cr_v16n_2x_noise.train.svm`；预期SHA-256=`2191fdc297aa49ce6a700df956ec44f68dc13e4c737937b31167543a203d22bb`，shape=(3897,75)，标签−1/+1。不要打开或重新hash历史source-test。

读取 `/root/feature-select/Radar-Ship-FS/experiments/k32_reward_robustness_v1/outer-partition.json` 与 `contexts/outer-fold-{0,1}.json`，只复用行号和映射。partition预期hash=`62235d2fde916ce00cb4fdea0224a08339c9ac26226c43f294074c6160e1fea6`。本轮只跑固定fold0/1，算法seed为`2026092401,2026092402`。无需全五折冻结仪式，但不能按成绩换掉某一折。

每fold在outer-train重算清洗，核对65列、clean 0-based与original 1-based坐标。图在outer-train构建：threshold=0.5、MI/tree seed=`20260915+fold`，`include_feature_id_node_feature=False`，domains全0。MI32按MI降序、同分original ID升序。保持全部合法候选可达。

J用outer-train内部`StratifiedKFold(3,shuffle=True,random_state=2026092201+fold)`的mean BAcc；每个inner fit独立拟合scaler。共同分类器：

```python
StandardScaler()
SVC(kernel="rbf", C=1.0, gamma=1/65, class_weight="balanced",
    probability=False, tol=1e-3, cache_size=256, max_iter=-1,
    shrinking=True)
```

outer评价也用同模型重拟合outer-train。All-LR另用StandardScaler+`LogisticRegression(C=1,solver="liblinear",class_weight="balanced",max_iter=5000,random_state=20260915+fold)`，不要把更换分类器的优势算成RL选择优势。原LRScorer是3×5及mean−0.5SD，不能拿来当新3-fold SVC scorer。

**可以逐case查看开发结果并据此提出一个变体。** 每个case的训练和archive只用inner J；case结束选好子集再算outer指标。不要每个checkpoint都用outer挑赢家。所有试过的版本、fold、seed和失败结果都保留即可，不需要首轮完成论文级统计。

## 4. 实现方法：成对重组PPO

### 4.1 环境与动作

每个状态S恰好32维，horizon=8。Pair-PPO动作是有序四元组`(r1,r2,a1,a2)`：r1/r2从动作前的S不放回选，a1/a2从动作前的补集不放回选，两个新增项不能是刚删除项。一步直接到`(S−{r1,r2})∪{a1,a2}`，只评价这个最终32维集合。**合法动作即执行，J下降也执行，不加贪心接受门。**

Single-PPO同框架，动作换为`(r,a)`。Random-Pair使用与Pair相同合法空间、环境、回报记录、重启和archive，但四个子动作均匀随机抽样。

### 4.2 条件化策略与PPO

复用或适配GraphActorCritic的64维编码；状态含selected mask、进度t/H、当前J及episode best J。pointer decoder按r1→r2→a1→a2生成，后一个head读取已选子动作embedding和当前图状态。可以新增薄适配类，不要求一次重构整个PPO包。Single沿用同类编码与两步decoder。

一次宏动作只有一个value/reward/transition，joint log-prob为各条件log-prob的和；PPO ratio对整个宏动作计算。更新时teacher-force同一有序动作序列和对应mask，不要把采样后排序的四元组代入旧log-prob。subset可排序作cache key，动作序列不能偷换。entropy取各有效子head熵的平均用于正则，避免Pair只因head多而获得双倍权重。

超参数：hidden64，Adam lr=3e-4、eps=1e-5；gamma=1、gae_lambda=1、clip=0.2、ppo_epochs=4、minibatch64、entropy_coef=0.01、value_coef=0.5、max_grad_norm=0.5、target_kl=0.05。每8个episodes更新一次，128训练episodes共16次更新；学习率先保持不变。关闭prior_scale、feature-ID通道/奖励、相关性惩罚、稀疏bonus和proxy shaping。

令b0=J(S0)，`b_t=max(b_(t−1),J(S_t))`，奖励`r_t=100×(b_t−b_(t−1))`。允许中间负向路径，奖励全路径找到的最好质量。保存J/current best曲线、非零奖励比例、PPO loss/entropy/梯度范数及训练前后参数变化；只做“确实训练”的基本检查，不先做完整因果归因。

### 4.3 重启与最终输出

三个搜索臂都遵循：

- 前16个训练episodes，MI32与随机32交替。
- 后续每4个episodes依次MI32、随机32、自身精英、自身精英。
- 精英为该臂已评分集合按J排序的top8不同子集，均匀取一个；同分按original ID元组排序。每个更新批次开始时固定archive快照，结束后统一纳入该批全部已评分集合。
- 随机起点由`default_rng(SeedSequence([seed,fold,20260922]))`预生成，每个对应episode在三臂相同；精英会随各臂结果不同，这是设计的一部分。动作与reset RNG分开记录，不需要不同算法耗用完全相同随机数。
- 128个训练episodes结束后冻结最后一个策略，不按outer选checkpoint；再跑32个episodes，起点只交替MI32/相同固定随机32，不用精英重启，不更新参数。

主endpoint是本case全部已评分集合中J最高者，包括训练和冻结rollout的中间状态及起点；另存冻结32条轨迹内的最好集合。不要要求后者先证明学习优势才能报告主endpoint的阳性。Random也保存相同两类结果。所有最终子集都输出两套坐标。

## 5. 少量对照，先把核心矩阵跑出来

基础矩阵：2fold×2seed×`[pair_ppo,single_ppo,random_pair]`=12个搜索case。每case128训练+32冻结episodes，每episode评分起点及8个新状态，共1440次J请求；随机臂也使用同样两段时长及重启切换。

每fold再运行一次：

- G1：完整exhaustive forward到32，允许forward分数下降但不提前停止；随后完整枚举1056个single-swap，最佳者J严格提高超过1e-12才接受，只做一轮。可复用已有纯函数。总2640请求，不要求先有“可优化空间”才运行RL。
- MI32-SVC、All-SVC，各一次J和outer评价；All-LR只作outer系统参照。

G1采用更多请求，是有意保留的强简单参照；首轮不声称等预算算法优势。三条搜索臂逻辑评分量相同。fold内确定性SVC缓存可共享以省实际时间，但不能把别的臂发现的子集/轨迹/精英传给当前臂。图和公共起点允许复用，记录实际拟合即可，不做繁琐摊销研究。

建议先跑两个fold的G1/MI/All，随后按`(0,seed1)→(1,seed1)→(0,seed2)→(1,seed2)`运行三臂成组case。各case完成即可评价、查看并记录，不必等整个矩阵冻结后才能看结果。

## 6. 预算、smoke与一次自主迭代

8个CPU评分worker，OMP/MKL/OpenBLAS/NumExpr各1线程，torch CPU threads=1。可用GPU可用于策略，分类器仍用CPU。并行收集8条episode，每轮批量评分，避免所有SVC拟合意外串行；不同时额外启动其他实验。

先用fold0训练区做≤100次classifier fits的smoke：实际SVC评分、合法宏动作、至少一次真实PPO更新、一个checkpoint保存/加载。smoke可缩成2个Pair episodes后做一次短更新，最多54次SVC fits；这不改变正式训练的8 episodes/update。smoke输出单独放，不并入正式成绩。没有SVC历史计时，按本次真实batch时间外推；若预算紧，优先完成一组完整三臂和其参照，再做下一组，不先花几个小时做审计。

预算算术：12×1440=17280请求，两个fold的G1/MI/All共5284请求，基础合计22564请求、最多67692次SVC折拟合。每fold图构建另有一次DT fit，端点评价/必要复核等都计入总物理classifier fits。

**总计算硬上限8小时、总物理classifier fits上限90000。目标一个工作日交付首批结果，动作适配复杂时允许到两个工作日。** 资源不足保存已完成成组对照及曲线，列出缺项，不把partial自动等同“方法不成立”。不为了凑齐审计文件拖延真实训练。

完成基础矩阵后，在余下预算内可以直接做 **一个** Pair-PPO修订，不用再次向用户确认：

- 若非零奖励比例低、曲线平：只把奖励换成`100×(J_new−J_old)`，仍允许降分动作；2fold×2seed、每case1440请求。
- 若已有阳性且轨迹仍有上升空间：只把horizon换16；训练64episodes、冻结16episodes、每4episodes更新，保持16次更新，总1360请求/case。

选择一个有曲线依据的修订，全部4个case从头训练，保存为独立variant；其他设置不变，不叠加改动，不隐去基础版。最多新增17280次SVC折拟合，基础+修订≤84972，剩余预算用于smoke、端点和修复。也允许判断无需修订而直接交付。若到硬预算则结束并报告，不无限扫描。

## 7. 新文件与入口

建议最少新增：

- `/root/feature-select/Radar-Ship-FS/src/run_block_rewrite.py`：train-only driver、SVC scorer、基线、调度、评价/报告。
- `/root/feature-select/Radar-Ship-FS/src/radar_ship_fs/ppo/block_rewrite.py`：宏动作模型/环境/buffer及PPO适配。
- `/root/feature-select/Radar-Ship-FS/configs/v16n/block_rewrite_ppo_v1.toml`。
- `/root/feature-select/Radar-Ship-FS/tests/test_block_rewrite.py`。
- `/root/feature-select/Radar-Ship-FS/documents/research-plan/08a-block-rewrite-results.md`。

正式输出 `/root/feature-select/Radar-Ship-FS/experiments/block_rewrite_ppo_v1/`，smoke输出同级`block_rewrite_ppo_smoke_v1/`。根据实现需要可合理拆文件，保持旧实验不变。配置+一次简短run note即可，不另造大篇幅冻结协议和审批流程。

下面是**待实现的新CLI契约**，不是既有命令；实现后核对--help。`run`执行基础矩阵，`refine`在配置中指定有理由的一次修订，`report`整理所有版本。

```bash
cd /root/feature-select/Radar-Ship-FS
export PYTHONPATH=src
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

/root/miniconda/envs/dl-lab/bin/python src/run_block_rewrite.py --help
/root/miniconda/envs/dl-lab/bin/python src/run_block_rewrite.py --config configs/v16n/block_rewrite_ppo_v1.toml --stage smoke --jobs 8
/root/miniconda/envs/dl-lab/bin/python src/run_block_rewrite.py --config configs/v16n/block_rewrite_ppo_v1.toml --stage run --jobs 8
# 有理由且预算允许时，配置variant后执行一次：
/root/miniconda/envs/dl-lab/bin/python src/run_block_rewrite.py --config configs/v16n/block_rewrite_ppo_v1.toml --stage refine --jobs 8
/root/miniconda/envs/dl-lab/bin/python src/run_block_rewrite.py --config configs/v16n/block_rewrite_ppo_v1.toml --stage report --jobs 8
```

smoke发现可修复技术问题，直接修复并记版本。case级checkpoint/resume应能继续工作；不为实现通用工作流框架推迟实验。

## 8. 最少检查与交付

先做足够保障训练有效的定向检查，再跑真实实验：

1. 两删两加合法、exact32、覆盖全部合法特征；子动作mask和条件log-prob正确，旧/新策略相同参数时PPO ratio≈1。
2. J下降的动作实际执行，best-so-far reward和有符号reward各自正确；episode边界与GAE不串轨迹。
3. 至少一次真实更新有有限梯度和参数变化；checkpoint能加载。禁用旧proxy和feature-ID奖励。
4. scaler/图不读取outer评价行，坐标映射正确，保存子集能重新计算分数；在一个正式endpoint做独立重算即可，不先复算所有结果。

实际运行`tests/test_block_rewrite.py`；若改共享模块，补相应已有测试。对新Python文件运行ruff。没有改生产代码的阶段无需全量pytest，不引用旧178 passed冒充当前通过。测试失败先修具体错误，不把排查扩大成全仓审计。

最低交付：配置与简短设计说明、真实命令/版本/数据身份、划分/映射、每个case的checkpoint及训练曲线、主与冻结策略子集、所有case的BAcc/accuracy/各类recall、配对差值、请求/拟合/墙钟、测试记录。建议保存`run.json`、`cases/`、`subsets.jsonl`、`results.csv`、`learning_curves.csv`、`costs.json`、`verification.json`和结果报告；不用凑齐上一版那套审计文件。

## 9. 判断：先发现，再解释

主关注Pair-PPO−G1的开发BAcc差，同时列Pair−Single、Pair−Random、Pair−MI、Pair−All-SVC；All-LR另列。每个fold先平均两个seed，再报两个fold均值与4个case原始差值；不要只展示最好seed，但允许把局部最好现象作为后续研究线索。

- 平均高于G1和Single：优先继续；+0.30～0.50 pp是有吸引力的量级，不是硬门槛。
- 只有某个完整case相对G1≥+0.50 pp：写“局部阳性，需再试”，允许继续投入，不因其余case波动否决。
- Random-Pair也好甚至更好：先承认成对搜索有空间，RL独立贡献待解释，不强行说策略学习成功，也不自动停止这条探索。
- 只提高inner J：保留为评分/过拟合问题的线索。
- 基础及一次合理修订均无可用信号：暂停这个具体块动作版本，后续唯一备选是LR下可变大小add/delete/STOP全局RL；本任务不顺手再启动第二条路线。

**取消**上一版的H_repeat、全五折、leave-one-fold、All/MI/LR全部过关、先无RL探测再GO的前置门槛。零更新消融、STG、精确预算匹配、真实场景确认、创新性最终论证放到阳性之后。

报告优先回答：“尝试了什么新机制？准确率最高到多少？相对哪些对照有阳性？值得下一次改哪一处？” 具体为什么阳性可以列待验证解释，不要求本轮证明完。不要编造成绩，不把开发调试后的最好结果包装成独立泛化证明。

完成实现、实际训练、有限修订及结果报告后结束。用户此次已经授权这个探索范围，不要再以‘需要先证明值得训练’为由退回规划。
