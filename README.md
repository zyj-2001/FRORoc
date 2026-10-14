
## 目录结构

```
FRORoc/
├── main.py            # 训练与测试主入口
├── model.py           # Model / CLEARProjection / FlowMatching / VelocityField
├── datahandler.py     # 数据加载
├── params.py          # 所有超参数定义
├── flow_matching/     # Flow path / scheduler / solver 库
├── utils/             # 日志与工具函数
└── datasets/
    ├── baby/
    ├── sports/
    └── tiktok/
```

每个数据集目录需包含：

```
trnMat.pkl        # 训练集用户-物品交互矩阵
tstMat.pkl        # 测试集
image_feat.npy    # 图像特征
text_feat.npy     # 文本特征
audio_feat.npy    # 音频特征（仅 tiktok）
```

---

## 环境依赖

```bash
pip install torch numpy scipy tqdm
```

---

## 快速运行

### tiktok（三模态，推荐配置）

```bash
python main.py \
  --data tiktok \
  --reg 1e-4 \
  --ssl_reg 1e-2 \
  --epoch 50 \
  --trans 1 \
  --e_loss 0.1 \
  --cl_method 1 \
  --scheduler_image cosine \
  --scheduler_text linear \
  --scheduler_audio vp
```

### baby / sports（双模态）

```bash
python main.py \
  --data baby \
  --reg 1e-4 \
  --ssl_reg 1e-2 \
  --epoch 50 \
  --trans 1 \
  --e_loss 0.1 \
  --cl_method 1 \
  --scheduler_image cosine \
  --scheduler_text linear
```

## 完整参数说明

### 基础参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data` | `allrecipes` | 数据集名称：`baby` / `sports` / `tiktok` |
| `--epoch` | `50` | 训练轮数 |
| `--lr` | `1e-3` | 学习率 |
| `--batch` | `1024` | 训练 batch 大小 |
| `--tstBat` | `256` | 测试 batch 大小 |
| `--reg` | `1e-5` | L2 正则权重 |
| `--latdim` | `64` | Embedding 维度 |
| `--gnn_layer` | `1` | GCN 层数 |
| `--topk` | `20` | 评测 top-K |
| `--gpu` | `0` | 使用的 GPU 编号 |
| `--seed` | `421` | 随机种子 |

### 对比学习参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--ssl_reg` | `1e-2` | 对比学习损失权重 |
| `--temp` | `0.5` | 对比学习温度 |
| `--cl_method` | `0` | `0`: 模态 vs 模态；`1`: 模态 vs 主干 |

### 模型结构参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--trans` | `0` | 特征投影方式：`0` 矩阵乘法，`1` Linear 层 |
| `--keepRate` | `0.5` | 图边保留比例（Dropout） |
| `--dims` | `[1000]` | VelocityField 隐层维度 |
| `--d_emb_size` | `10` | 时间步嵌入维度 |
| `--ris_lambda` | `0.5` | 模态图残差融合权重 |
| `--ris_adj_lambda` | `0.2` | 重建邻接矩阵融合权重 |
| `--rebuild_k` | `1` | 每用户重建 top-K 连边数 |
| `--e_loss` | `0.1` | 图对比损失权重 |

### Flow Matching 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--scheduler` | `cosine` | 全局 fallback scheduler |
| `--scheduler_image` | `None`（用 scheduler） | image 模态的 flow path |
| `--scheduler_text` | `None`（用 scheduler） | text 模态的 flow path |
| `--scheduler_audio` | `None`（用 scheduler） | audio 模态的 flow path |
| `--ode_method` | `midpoint` | ODE 求解器：`euler` / `midpoint` / `rk4` |
| `--steps` | `10` | ODE 积分步数 |
| `--noise_scale` | `0.1` | 初始噪声标准差 |
| `--use_ot` | `False` | 开启 Minibatch OT 配对（需要 scipy） |

**scheduler 可选值**：`cosine` | `linear` | `vp` | `linear_vp` | `poly2` | `poly3`

### ROP 去冗余参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--clear_rank` | `0`（关闭） | SVD 共享方向数 r，推荐搜索 `4 / 8 / 16 / 32` |
| `--clear_lam` | `1.0` | 投影强度 λ，`1.0` = 完全去除共享子空间 |
| `--clear_text` | `False` | 同时对 text 做投影（默认 text-first：只投影 image） |

