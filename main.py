# ml_backend/main.py
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union, Literal
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score

from config import DATASET_ROOTS, OMICS_PREFIX, DEFAULT_PROJECTS_VAL

# ==== App & CORS =============================================================
app = FastAPI(title="Cancer ML API")

# 同源托管前端其实不需要 CORS，但保留宽松设置便于调试
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==== Part 2: 经典 ML =======================================================
# —— 模型池 ——
MODEL_POOL = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "SVM": SVC(),
    "RandomForest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "RidgeClassifier": RidgeClassifier(),
    "NaiveBayes": GaussianNB(),
}

# —— Pydantic 请求/响应（Part 2） ——
class RunRequest(BaseModel):
    project: str
    omics_combos: List[List[str]]      # 例: [["1"],["1","2","3"]]
    models: List[str]                  # 例: ["SVM","RandomForest"]
    dataset_root_key: str = "val"      # "val" | "pretrain"

class RunResponseRow(BaseModel):
    project: str
    omics: str
    model: str
    acc: float
    f1w: float
    f1m: float
    n_train: int
    n_test: int

class RunResponse(BaseModel):
    rows: List[RunResponseRow]

# —— 工具函数（Part 2） ——
def list_projects_by_root(root_key: str) -> List[str]:
    """根据根目录自动扫描项目文件夹名。"""
    root = DATASET_ROOTS.get(root_key)
    if not root:
        raise HTTPException(status_code=400, detail=f"Unknown dataset root: {root_key}")
    if not root.exists():
        raise HTTPException(status_code=404, detail=f"Dataset root not found: {root}")
    # 只返回文件夹名
    return sorted([p.name for p in root.iterdir() if p.is_dir()])

def load_data(project_dir: Path, omic_code: str):
    tr = pd.read_csv(project_dir / f"{omic_code}_tr.csv", index_col=0)
    te = pd.read_csv(project_dir / f"{omic_code}_te.csv", index_col=0)
    return tr, te

def load_labels(project_dir: Path):
    y_tr = pd.read_csv(project_dir / "labels_tr.csv")
    y_te = pd.read_csv(project_dir / "labels_te.csv")
    y_tr = y_tr.drop_duplicates(subset="pid").set_index("pid").loc[:, "Label"]
    y_te = y_te.drop_duplicates(subset="pid").set_index("pid").loc[:, "Label"]
    return y_tr, y_te

def evaluate_model(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    acc = float(accuracy_score(y_test, pred))
    f1w = float(f1_score(y_test, pred, average="weighted"))
    f1m = float(f1_score(y_test, pred, average="macro"))
    return acc, f1w, f1m

# —— 元数据：返回根键/模型/组合选项（不含项目列表） ——
@app.get("/datasets")
def list_datasets():
    return {
        "dataset_roots": list(DATASET_ROOTS.keys()),  # ["val","pretrain"]
        "models": list(MODEL_POOL.keys()),
        "omics_options": [
            {"code": "1", "name": OMICS_PREFIX["1"]},
            {"code": "2", "name": OMICS_PREFIX["2"]},
            {"code": "3", "name": OMICS_PREFIX["3"]},
        ],
        "default_combos": [
            ["1"], ["2"], ["3"],
            ["1","2"], ["1","3"], ["2","3"],
            ["1","2","3"]
        ],
    }

# —— 新增接口：按根目录获取项目列表 ——
@app.get("/projects")
def get_projects(root: str = Query(..., description="dataset root key: val | pretrain")):
    try:
        return {"root": root, "projects": list_projects_by_root(root)}
    except HTTPException as e:
        raise e

# —— 运行 ML ——
@app.post("/run-ml", response_model=RunResponse)
def run_ml(req: RunRequest):
    if req.dataset_root_key not in DATASET_ROOTS:
        raise HTTPException(status_code=400, detail=f"Unknown dataset root key: {req.dataset_root_key}")

    # 检查模型名
    for m in req.models:
        if m not in MODEL_POOL:
            raise HTTPException(status_code=400, detail=f"Unknown model: {m}")

    project_dir = DATASET_ROOTS[req.dataset_root_key] / req.project
    if not project_dir.exists():
        raise HTTPException(status_code=404, detail=f"Project folder not found: {project_dir}")

    y_tr, y_te = load_labels(project_dir)

    results: List[Dict[str, Any]] = []
    for combo in req.omics_combos:
        try:
            x_tr_parts, x_te_parts = [], []
            for om in combo:
                tr, te = load_data(project_dir, om)
                x_tr_parts.append(tr)
                x_te_parts.append(te)
            x_tr = pd.concat(x_tr_parts, axis=1)
            x_te = pd.concat(x_te_parts, axis=1)

            # 对齐
            common_tr = x_tr.index.intersection(y_tr.index)
            common_te = x_te.index.intersection(y_te.index)
            x_tr = x_tr.loc[common_tr]
            y_tr_aligned = y_tr.loc[common_tr]
            x_te = x_te.loc[common_te]
            y_te_aligned = y_te.loc[common_te]

            # 标准化
            scaler = StandardScaler()
            x_tr_scaled = scaler.fit_transform(x_tr)
            x_te_scaled = scaler.transform(x_te)

            for model_name in req.models:
                model = MODEL_POOL[model_name]
                acc, f1w, f1m = evaluate_model(model, x_tr_scaled, y_tr_aligned, x_te_scaled, y_te_aligned)
                results.append({
                    "project": req.project,
                    "omics": "+".join(combo),
                    "model": model_name,
                    "acc": acc,
                    "f1w": f1w,
                    "f1m": f1m,
                    "n_train": int(x_tr.shape[0]),
                    "n_test": int(x_te.shape[0]),
                })
        except Exception as e:
            results.append({
                "project": req.project,
                "omics": "+".join(combo),
                "model": "—",
                "acc": np.nan, "f1w": np.nan, "f1m": np.nan,
                "n_train": 0, "n_test": 0,
                "error": str(e)
            })
    return {"rows": results}

@app.get("/health")
def health():
    return {"ok": True}


# ==== Part 3: 深度学习（DL） =================================================
# 说明：
# - 按真实数据文件加载，不再使用占位的随机数据；
# - 支持 linear / simple_mlp / attention；
# - 训练遍历所选 omics 组合，返回完整结果表 rows；
# - epoch/epochs 实时上报用于前端进度条。

import uuid, json, hashlib
from typing import Optional, List, Dict, Any, Union, Literal

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from models_nn import build_model  # 与你的项目结构一致
import numpy as _np


_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_JOBS_DL: Dict[str, Dict[str, Any]] = {}

def _grad_importance_all_features(model: nn.Module, val_loader: DataLoader, device: torch.device) -> _np.ndarray:
    """
    对验证集计算输入特征的梯度重要性：
      score_j = mean(|∂logit_true/∂x_j|)
    返回长度为总特征数的 numpy 数组。
    """
    model.eval()
    feat_sum = None
    n_total = 0
    for xb, yb in val_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        xb.requires_grad_(True)

        logits = model(xb)
        chosen = logits.gather(1, yb.view(-1, 1)).sum()

        model.zero_grad(set_to_none=True)
        if xb.grad is not None:
            xb.grad.zero_()
        chosen.backward()

        g = xb.grad.detach().abs().sum(dim=0)  # 按 batch 求和
        feat_sum = g if feat_sum is None else (feat_sum + g)
        n_total += xb.size(0)

        xb.requires_grad_(False)

    imp = (feat_sum / max(n_total, 1)).detach().cpu().numpy()
    return imp


import re  # 顶部若尚未引入

def _parse_combo_str(combo: Optional[str]) -> List[str]:
    """
    兼容 '1+2'、'1,2'、'1 2'；空/ALL => ['1','2','3']。
    """
    if combo is None:
        return ["1", "2", "3"]
    s = str(combo).strip()
    if not s or s.upper() == "ALL":
        return ["1", "2", "3"]
    parts = re.split(r"[+,;|\s]+", s)
    codes = [p for p in parts if p in {"1", "2", "3"}]
    return codes or ["1", "2", "3"]


def _load_combo_dataframe(project_dir: Path, combo: Optional[str]):
    """
    读取一个 omics 组合（如 '1+2' 或 '1,2'）并与标签对齐。
    返回 6 项：
      x_tr_df, x_te_df, y_tr_s, y_te_s, mrna_slice, mrna_featnames
        - mrna_slice: (start, end) 在拼接后的列索引范围；若不含 '1' 则为 None
        - mrna_featnames: 从 1_featname.csv 读到的 ENSG 列表；若不存在则为空
    """
    codes = _parse_combo_str(combo)
    x_tr_parts, x_te_parts = [], []

    total_cols = 0
    mrna_slice = None
    mrna_featnames: List[str] = []

    for om in codes:
        tr = pd.read_csv(project_dir / f"{om}_tr.csv", index_col=0)
        te = pd.read_csv(project_dir / f"{om}_te.csv", index_col=0)

        # 记录 mRNA 的列范围与 featname
        if om == "1":
            mrna_slice = (total_cols, total_cols + tr.shape[1])
            feat_path = project_dir / "1_featname.csv"
            if feat_path.exists():
                fn = pd.read_csv(feat_path, header=None)
                mrna_featnames = fn.iloc[:, 0].astype(str).tolist()

        x_tr_parts.append(tr)
        x_te_parts.append(te)
        total_cols += tr.shape[1]

    x_tr_df = pd.concat(x_tr_parts, axis=1)
    x_te_df = pd.concat(x_te_parts, axis=1)

    # 标签
    y_tr = pd.read_csv(project_dir / "labels_tr.csv")
    y_te = pd.read_csv(project_dir / "labels_te.csv")
    y_tr_s = y_tr.drop_duplicates(subset="pid").set_index("pid").loc[:, "Label"]
    y_te_s = y_te.drop_duplicates(subset="pid").set_index("pid").loc[:, "Label"]

    # 对齐
    common_tr = x_tr_df.index.intersection(y_tr_s.index)
    common_te = x_te_df.index.intersection(y_te_s.index)
    x_tr_df = x_tr_df.loc[common_tr]
    y_tr_s  = y_tr_s.loc[common_tr]
    x_te_df = x_te_df.loc[common_te]
    y_te_s  = y_te_s.loc[common_te]

    return x_tr_df, x_te_df, y_tr_s, y_te_s, mrna_slice, mrna_featnames


# ---------- PCA & 标准化 ----------
def _pca2d(tensor_X: torch.Tensor, max_points: int = 300):
    X = tensor_X.detach().cpu().float().numpy()[:max_points]
    X = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = _np.linalg.svd(X, full_matrices=False)
    Z = U[:, :2] * S[:2]
    return Z  # [N',2]

def _standardize_trainval(Xtr: torch.Tensor, Xv: torch.Tensor):
    mu = Xtr.mean(dim=0, keepdim=True)
    std = Xtr.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    return (Xtr - mu) / std, (Xv - mu) / std

# ---------- Pydantic 配置 ----------
class _BaseCfg(BaseModel):
    name: str = "MyDLRun"
    input_dim: int
    num_classes: int
    seed: Optional[int] = 42

class _LinearCfg(_BaseCfg):
    kind: Literal["linear"]

class _MLPCfg(_BaseCfg):
    kind: Literal["simple_mlp"]
    hidden_dims: List[int] = [256, 128, 64]
    dropout: float = 0.2
    activation: Literal["relu", "gelu", "elu", "leaky_relu"] = "relu"
    batch_norm: bool = True

class _AttCfg(_BaseCfg):
    kind: Literal["attention"]
    d_model: int = 32
    n_heads: int = 2
    n_layers: int = 2
    ff_multiplier: int = 2
    dropout: float = 0.1
    max_len: Optional[int] = None

AnyCfg = Union[_LinearCfg, _MLPCfg, _AttCfg]

class _TrainCfgDL(BaseModel):
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    epochs: int = 200
    balanced: bool = True
    standardize: bool = True   # 默认做特征标准化（对齐 baseline）

class DLTrainRequest(BaseModel):
    # 允许强类型或裸 dict（避免前端字段变动时 422）
    model: Union[AnyCfg, Dict[str, Any]]
    train: _TrainCfgDL
    dataset_id: str                 # 例：TCGA-UCS
    dataset_root_key: str = "val"   # 例：val / pretrain
    omics_combos: Optional[List[str]] = None  # 例：["1","1+2","1+2+3"]

@app.get("/model/kinds")
def dl_kinds():
    return {"kinds": ["linear", "simple_mlp", "attention"]}

import re  # ← 若顶部没有，请添加

def _parse_combo_str(combo: Optional[str]) -> List[str]:
    """
    把组合字符串解析为代码列表，兼容 '1+2'、'1,2'、'1 2' 等写法。
    为空或 'ALL' 时返回 ['1','2','3']。
    """
    if combo is None:
        return ["1", "2", "3"]
    s = str(combo).strip()
    if not s or s.upper() == "ALL":
        return ["1", "2", "3"]
    parts = re.split(r"[+,;|\s]+", s)
    codes = [p for p in parts if p in {"1", "2", "3"}]
    return codes or ["1", "2", "3"]


def _dl_set_seed(seed: Optional[int]):
    if seed is None:
        return
    import random, numpy as _np, torch as _torch
    random.seed(seed)
    _np.random.seed(seed)
    _torch.manual_seed(seed)
    _torch.cuda.manual_seed_all(seed)




def _ensure_long_labels(y_tr_s: pd.Series, y_te_s: pd.Series):
    """
    把 Series 标签统一转成 torch.long（CrossEntropyLoss 的 target）。
    - 若本身是数字，直接转；
    - 若是字符串/分类，做一个全局映射（train+val 的并集）。
    返回：y_tr_t, y_te_t, num_classes
    """
    if np.issubdtype(y_tr_s.dtype, np.number) and np.issubdtype(y_te_s.dtype, np.number):
        y_tr_t = torch.tensor(y_tr_s.astype(int).to_numpy(), dtype=torch.long)
        y_te_t = torch.tensor(y_te_s.astype(int).to_numpy(), dtype=torch.long)
        num_classes = int(max(y_tr_t.max().item(), y_te_t.max().item()) + 1)
        return y_tr_t, y_te_t, num_classes

    # 字符串/分类：做映射
    all_cats = pd.concat([y_tr_s.astype(str), y_te_s.astype(str)], axis=0).unique().tolist()
    idx_map = {c: i for i, c in enumerate(all_cats)}
    y_tr_t = torch.tensor([idx_map[str(v)] for v in y_tr_s.tolist()], dtype=torch.long)
    y_te_t = torch.tensor([idx_map[str(v)] for v in y_te_s.tolist()], dtype=torch.long)
    num_classes = len(all_cats)
    return y_tr_t, y_te_t, num_classes

def _dl_load_dataset_files(root_key: str, dataset_id: str, combo: Optional[str]):
    """
    严格按文件读取一个组合：
    返回 torch 张量：Xtr, ytr, Xv, yv
    """
    if root_key not in DATASET_ROOTS:
        raise HTTPException(status_code=400, detail=f"Unknown dataset root key: {root_key}")
    project_dir = DATASET_ROOTS[root_key] / dataset_id
    if not project_dir.exists():
        raise HTTPException(status_code=404, detail=f"Project folder not found: {project_dir}")

    x_tr_df, x_te_df, y_tr_s, y_te_s = _load_combo_dataframe(project_dir, combo)
    # 转 tensor
    Xtr = torch.tensor(x_tr_df.to_numpy(dtype=np.float32), dtype=torch.float32)
    Xv  = torch.tensor(x_te_df.to_numpy(dtype=np.float32), dtype=torch.float32)
    ytr, yv, num_classes = _ensure_long_labels(y_tr_s, y_te_s)
    return Xtr, ytr, Xv, yv, num_classes

# ---------- 训练主流程：遍历所有组合并返回完整表 ----------
@app.post("/model/train")
def dl_train(req: DLTrainRequest):
    job_id = str(uuid.uuid4())
    _JOBS_DL[job_id] = {"status": "queued", "step": "queued"}

    try:
        # 读取配置（兼容 pydantic v1/v2 + dict）
        if isinstance(req.model, dict):
            base_model_cfg = req.model
        else:
            base_model_cfg = req.model.model_dump() if hasattr(req.model, "model_dump") else req.model.dict()
        train_cfg = req.train.model_dump() if hasattr(req.train, "model_dump") else req.train.dict()

        per_combo_epochs = int(train_cfg.get("epochs", 20))
        combos: List[Optional[str]] = (req.omics_combos or [])
        if not combos:
            combos = [None]  # 没传则当作 ALL（1+2+3）
        total_epochs = per_combo_epochs * len(combos)

        rows: List[Dict[str, Any]] = []
        biomarkers: List[Dict[str, Any]] = []   # ← 新增：每个组合的 mRNA Top50
        best_overall = -1.0

        _dl_set_seed(base_model_cfg.get("seed", 42))

        # 静态下载目录（通过 /downloads/... 可直接访问）
        out_static_dir = (STATIC_DIR / "downloads" / job_id)
        out_static_dir.mkdir(parents=True, exist_ok=True)

        # —— 逐组合训练 ——
        global_epoch = 0
        for combo_idx, combo in enumerate(combos, start=1):
            combo_disp = "+".join(_parse_combo_str(combo))

            # 严格复用 Part2 读取 + 对齐，并拿到 mRNA 切片与 ENSG 名称
            root_key = getattr(req, "dataset_root_key", "val")
            project_dir = DATASET_ROOTS[root_key] / req.dataset_id
            if not project_dir.exists():
                raise HTTPException(status_code=404, detail=f"Project folder not found: {project_dir}")

            x_tr_df, x_te_df, y_tr_s, y_te_s, mrna_slice, mrna_featnames = _load_combo_dataframe(project_dir, combo)

            # 转成张量
            Xtr = torch.tensor(x_tr_df.values, dtype=torch.float32)
            Xv  = torch.tensor(x_te_df.values, dtype=torch.float32)
            # 统一把标签映射到 0..K-1（字符串也可）
            classes = sorted(set(map(str, y_tr_s.tolist() + y_te_s.tolist())))
            cls2id = {c: i for i, c in enumerate(classes)}
            ytr = torch.tensor([cls2id[str(v)] for v in y_tr_s.tolist()], dtype=torch.long)
            yv  = torch.tensor([cls2id[str(v)] for v in y_te_s.tolist()], dtype=torch.long)
            num_classes = len(classes)

            # 以真实数据维度/类别数为准
            model_cfg = dict(base_model_cfg)
            model_cfg["input_dim"] = int(Xtr.shape[1])
            model_cfg["num_classes"] = int(num_classes)

            # 标准化（默认 True）
            if bool(train_cfg.get("standardize", True)):
                Xtr, Xv = _standardize_trainval(Xtr, Xv)

            # DataLoader
            bs = int(train_cfg.get("batch_size", 64))
            train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=bs, shuffle=True)
            val_loader   = DataLoader(TensorDataset(Xv,  yv ), batch_size=256, shuffle=False)

            # 模型与优化器
            model = build_model(model_cfg).to(_DEVICE)
            criterion = nn.CrossEntropyLoss()
            if model_cfg.get("kind") == "linear":
                optimizer = torch.optim.Adam(model.parameters(), lr=float(train_cfg.get("lr", 5e-4)))
            else:
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=float(train_cfg.get("lr", 1e-3)),
                    weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
                )

            # 训练该组合
            best_val = -1.0
            for ep in range(1, per_combo_epochs + 1):
                global_epoch += 1
                model.train()
                for xb, yb in train_loader:
                    xb, yb = xb.to(_DEVICE), yb.to(_DEVICE)
                    optimizer.zero_grad()
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    loss.backward()
                    optimizer.step()

                # 验证（用于进度展示 acc）
                model.eval()
                correct, total = 0, 0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb, yb = xb.to(_DEVICE), yb.to(_DEVICE)
                        pred = model(xb).argmax(dim=1)
                        correct += int((pred == yb).sum().item())
                        total += len(xb)
                val_acc = correct / max(total, 1)
                best_val = max(best_val, val_acc)

                _JOBS_DL[job_id] = {
                    "status": "running",
                    "step": f"combo {combo_idx}/{len(combos)}",
                    "epoch": global_epoch,
                    "epochs": total_epochs,
                    "metrics": {"val_acc": val_acc},
                    "current_combo": combo_disp,
                }

            # —— 组合结束：完整指标（acc/f1w/f1m） —— #
            from sklearn.metrics import accuracy_score, f1_score
            y_true_list, y_pred_list = [], []
            model.eval()
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(_DEVICE)
                    logits = model(xb)
                    y_pred_list.append(logits.argmax(dim=1).cpu())
                    y_true_list.append(yb.cpu())
            y_true = torch.cat(y_true_list).numpy()
            y_pred = torch.cat(y_pred_list).numpy()
            acc = float(accuracy_score(y_true, y_pred))
            f1w = float(f1_score(y_true, y_pred, average="weighted"))
            f1m = float(f1_score(y_true, y_pred, average="macro"))

            rows.append({
                "project": req.dataset_id,
                "omics": combo_disp,
                "model": model_cfg.get("kind", "mlp"),
                "acc": acc,
                "f1w": f1w,
                "f1m": f1m,
                "n_train": int(Xtr.shape[0]),
                "n_test": int(Xv.shape[0]),
            })
            best_overall = max(best_overall, best_val)

            # —— mRNA 贡献度（只在组合里含 '1' 时计算）+ 保存 CSV —— #
            if mrna_slice is not None:
                imp_all = _grad_importance_all_features(model, val_loader, _DEVICE)  # [F]
                s, e = mrna_slice
                mrna_scores = imp_all[s:e] if e > s else _np.array([])

                # 名称：优先用 1_featname.csv；长度不匹配则退化为列序号
                if len(mrna_featnames) == (e - s):
                    names = mrna_featnames
                else:
                    names = [f"mrna_{i}" for i in range(e - s)]

                if mrna_scores.size > 0:
                    order = _np.argsort(-mrna_scores)[:50]
                    top = []
                    for r, idx in enumerate(order, start=1):
                        top.append({
                            "rank": int(r),
                            "feature": str(names[idx]),
                            "score": float(mrna_scores[idx]),
                        })
                    # 保存 CSV 到静态目录
                    df_top = pd.DataFrame(top)
                    fname = f"biomarkers_mrna_combo-{combo_disp.replace('+','-')}.csv"
                    df_top.to_csv(out_static_dir / fname, index=False, encoding="utf-8")
                    biomarkers.append({
                        "combo": combo_disp,
                        "csv": f"/downloads/{job_id}/{fname}",
                        "top50": top,
                    })
                else:
                    biomarkers.append({
                        "combo": combo_disp,
                        "csv": None,
                        "top50": [],
                    })

        # —— 完成 —— #
        outdir = Path("results") / job_id
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "metrics.json").write_text(json.dumps({"best_val_acc": best_overall}), encoding="utf-8")

        _JOBS_DL[job_id] = {
            "status": "succeeded",
            "step": "done",
            "epoch": total_epochs,
            "epochs": total_epochs,
            "metrics": {"best_val_acc": best_overall},
            "rows": rows,
            "biomarkers": biomarkers,   # ← 前端用这个渲染 Top50
            "selected_omics": req.omics_combos or None,
        }

    except Exception as e:
        _JOBS_DL[job_id] = {"status": "failed", "message": str(e)}

    return {"job_id": job_id, "status": _JOBS_DL[job_id]["status"]}





@app.get("/model/train/{job_id}")
def dl_progress(job_id: str):
    return _JOBS_DL.get(job_id, {"status": "failed", "message": "not found"})

# ==== 静态托管前端 ===========================================================
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
else:
    @app.get("/")
    def root():
        return {"ok": True, "msg": "Static UI not found. API is running."}


