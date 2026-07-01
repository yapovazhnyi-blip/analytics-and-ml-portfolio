"""
XGBoost and LightGBM model families for Crucible's AutoML pipeline.

WHY THESE TWO MATTER
--------------------
The five existing sklearn families (Random Forest, Gradient Boosting,
Logistic Regression, SVM, k-NN) cover the conceptual landscape well.
XGBoost and LightGBM are not conceptually different — they are both
gradient-boosted tree ensembles — but they are dramatically better
implementations of the same idea:

  sklearn GradientBoostingClassifier: pure Python, no parallelism,
    grows trees level-by-level. Slow on large datasets.

  XGBoost: C++ core, fully parallel, regularisation (L1/L2) built into
    the split criterion, handles missing values natively, supports
    GPU training. Typically 10–100× faster than sklearn GBT.

  LightGBM: Microsoft's implementation, adds two key innovations:
    - GOSS (Gradient-based One-Side Sampling): only computes gradients
      on samples with large residuals, ignoring well-fitted samples.
      Reduces data used per round without losing accuracy.
    - EFB (Exclusive Feature Bundling): combines sparse features that
      rarely have non-zero values simultaneously, reducing feature count.
    Consequence: LightGBM is often faster than XGBoost and uses less
    memory, with similar accuracy. Leaf-wise growth (vs XGBoost's
    level-wise) produces deeper trees with fewer estimators.

Both dominate Kaggle tabular competitions and production ML systems.
Their absence from a portfolio AutoML tool is immediately noticed by
any practitioner interviewer.

HYPERPARAMETER SPACES
---------------------
The spaces are designed to be broad enough to capture the meaningful
range of each parameter while avoiding the tails that almost never win:

  n_estimators:     100–800 (log scale)  — too few and the model
                    underfits; too many and training time dominates
                    without accuracy gains. log scale because the
                    difference between 100 and 200 matters more than
                    the difference between 700 and 800.

  max_depth:        3–9 (XGBoost, linear)  — shallow trees generalise
  max_depth:        3–12 (LightGBM, linear) — LightGBM uses leaf-wise
  num_leaves:       20–300 (LightGBM, log) — more meaningful than depth
                    for leaf-wise trees; constrains complexity directly.

  learning_rate:    0.01–0.3 (log) — the most important parameter after
                    n_estimators. Lower LR needs more trees to converge
                    but generalises better. Log scale because 0.01 and
                    0.05 behave very differently while 0.25 and 0.29 do not.

  subsample:        0.6–1.0 — fraction of rows sampled per tree.
  colsample_bytree: 0.6–1.0 — fraction of columns sampled per tree.
                    Both add stochasticity that reduces overfitting.
                    Below 0.6 usually hurts too much.

  reg_alpha (L1):   1e-8–10.0 (log) — drives weights toward zero,
                    produces sparse models, useful when many features
                    are irrelevant.
  reg_lambda (L2):  1e-8–10.0 (log) — shrinks weights smoothly,
                    generally safer default regularisation.

OPTIONAL IMPORT
---------------
Both libraries are listed in requirements.txt but failures here should
not crash the rest of the AutoML pipeline. XGBOOST_AVAILABLE and
LIGHTGBM_AVAILABLE flags allow model_families.py to register these
families only when the libraries are present.
"""

from __future__ import annotations

from typing import Optional

# ── Optional imports ───────────────────────────────────────────────────────

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


# ── XGBoost families ───────────────────────────────────────────────────────

def xgb_classifier(trial):
    """
    XGBoost classifier with Optuna hyperparameter search.

    Returns a bare XGBClassifier (no Pipeline wrapper) since XGBoost is
    tree-based and insensitive to feature scaling. The TrainingRunner
    handles cross-validation exactly as it does for the sklearn families.

    XGBoost 2+ no longer requires use_label_encoder=False — omitted.
    verbosity=0 suppresses the per-round training log that would otherwise
    flood stdout during the Optuna search.
    """
    if not XGBOOST_AVAILABLE:
        raise ImportError("xgboost is required. Run: pip install xgboost")

    return XGBClassifier(
        n_estimators    = trial.suggest_int  ("xgb_clf_n_estimators",    100,  800, log=True),
        max_depth       = trial.suggest_int  ("xgb_clf_max_depth",         3,    9),
        learning_rate   = trial.suggest_float("xgb_clf_lr",             0.01,  0.3, log=True),
        subsample       = trial.suggest_float("xgb_clf_subsample",       0.6,  1.0),
        colsample_bytree= trial.suggest_float("xgb_clf_colsample_bytree",0.6,  1.0),
        min_child_weight= trial.suggest_int  ("xgb_clf_min_child_weight",  1,    7),
        gamma           = trial.suggest_float("xgb_clf_gamma",           0.0,  1.0),
        reg_alpha       = trial.suggest_float("xgb_clf_reg_alpha",      1e-8, 10.0, log=True),
        reg_lambda      = trial.suggest_float("xgb_clf_reg_lambda",     1e-8, 10.0, log=True),
        random_state=42,
        verbosity=0,
        n_jobs=-1,
    )


def xgb_regressor(trial):
    """XGBoost regressor — same hyperparameter space as the classifier."""
    if not XGBOOST_AVAILABLE:
        raise ImportError("xgboost is required. Run: pip install xgboost")

    return XGBRegressor(
        n_estimators    = trial.suggest_int  ("xgb_reg_n_estimators",    100,  800, log=True),
        max_depth       = trial.suggest_int  ("xgb_reg_max_depth",         3,    9),
        learning_rate   = trial.suggest_float("xgb_reg_lr",             0.01,  0.3, log=True),
        subsample       = trial.suggest_float("xgb_reg_subsample",       0.6,  1.0),
        colsample_bytree= trial.suggest_float("xgb_reg_colsample_bytree",0.6,  1.0),
        min_child_weight= trial.suggest_int  ("xgb_reg_min_child_weight",  1,    7),
        gamma           = trial.suggest_float("xgb_reg_gamma",           0.0,  1.0),
        reg_alpha       = trial.suggest_float("xgb_reg_alpha",          1e-8, 10.0, log=True),
        reg_lambda      = trial.suggest_float("xgb_reg_lambda",         1e-8, 10.0, log=True),
        random_state=42,
        verbosity=0,
        n_jobs=-1,
    )


# ── LightGBM families ──────────────────────────────────────────────────────

def lgbm_classifier(trial):
    """
    LightGBM classifier with Optuna hyperparameter search.

    num_leaves is more meaningful than max_depth for LightGBM because
    it grows trees leaf-wise (best-first) rather than level-wise. A tree
    with max_depth=6 has at most 64 leaves, but leaf-wise growth with
    num_leaves=64 can achieve higher accuracy with fewer splits overall.

    subsample_freq=1 is required for the subsample parameter to actually
    take effect — LightGBM only applies bagging when subsample_freq > 0.

    verbose=-1 suppresses LightGBM's per-iteration training log.
    """
    if not LIGHTGBM_AVAILABLE:
        raise ImportError("lightgbm is required. Run: pip install lightgbm")

    return LGBMClassifier(
        n_estimators     = trial.suggest_int  ("lgbm_clf_n_estimators",     100,  800, log=True),
        max_depth        = trial.suggest_int  ("lgbm_clf_max_depth",           3,   12),
        learning_rate    = trial.suggest_float("lgbm_clf_lr",              0.01,  0.3, log=True),
        num_leaves       = trial.suggest_int  ("lgbm_clf_num_leaves",        20,  300, log=True),
        min_child_samples= trial.suggest_int  ("lgbm_clf_min_child_samples",   5,  100),
        subsample        = trial.suggest_float("lgbm_clf_subsample",        0.6,  1.0),
        subsample_freq   = 1,
        colsample_bytree = trial.suggest_float("lgbm_clf_colsample_bytree", 0.6,  1.0),
        reg_alpha        = trial.suggest_float("lgbm_clf_reg_alpha",       1e-8, 10.0, log=True),
        reg_lambda       = trial.suggest_float("lgbm_clf_reg_lambda",      1e-8, 10.0, log=True),
        random_state=42,
        verbose=-1,
        n_jobs=-1,
    )


def lgbm_regressor(trial):
    """LightGBM regressor — same hyperparameter space as the classifier."""
    if not LIGHTGBM_AVAILABLE:
        raise ImportError("lightgbm is required. Run: pip install lightgbm")

    return LGBMRegressor(
        n_estimators     = trial.suggest_int  ("lgbm_reg_n_estimators",     100,  800, log=True),
        max_depth        = trial.suggest_int  ("lgbm_reg_max_depth",           3,   12),
        learning_rate    = trial.suggest_float("lgbm_reg_lr",              0.01,  0.3, log=True),
        num_leaves       = trial.suggest_int  ("lgbm_reg_num_leaves",        20,  300, log=True),
        min_child_samples= trial.suggest_int  ("lgbm_reg_min_child_samples",   5,  100),
        subsample        = trial.suggest_float("lgbm_reg_subsample",        0.6,  1.0),
        subsample_freq   = 1,
        colsample_bytree = trial.suggest_float("lgbm_reg_colsample_bytree", 0.6,  1.0),
        reg_alpha        = trial.suggest_float("lgbm_reg_alpha",           1e-8, 10.0, log=True),
        reg_lambda       = trial.suggest_float("lgbm_reg_lambda",          1e-8, 10.0, log=True),
        random_state=42,
        verbose=-1,
        n_jobs=-1,
    )


# ── CatBoost ──────────────────────────────────────────────────────────────────
#
# WHY CATBOOST MATTERS
# --------------------
# CatBoost (Yandex, 2017) is the third pillar of gradient-boosted trees alongside
# XGBoost and LightGBM. Its key advantage for fintech and data-science roles:
#
#   Ordered boosting — uses a permutation-based algorithm to prevent target
#     leakage during training. Standard GBM computes leaf statistics on the
#     same data used to build the tree, which causes overfitting on small
#     datasets. CatBoost avoids this without requiring a separate validation split.
#
#   Native categorical handling — applies ordered target encoding internally
#     without the user having to pre-encode categoricals. On high-cardinality
#     columns (merchant codes, geography, product categories — all common in
#     fintech) CatBoost consistently outperforms XGBoost and LightGBM when
#     those columns aren't carefully pre-encoded.
#
#   Symmetric (oblivious) trees — uses the same split condition at every
#     node of a given depth level, making inference O(depth) instead of O(nodes).
#     Deployed models are significantly faster at inference time.
#
# The availability guard follows the same pattern as XGBoost and LightGBM.

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


def catboost_classifier(trial):
    """
    CatBoost classifier with Optuna search over its key hyperparameters.

    CatBoost is particularly effective for:
      - High-cardinality categorical features (native encoding)
      - Small-to-medium datasets (ordered boosting prevents overfitting)
      - Production inference (symmetric trees are fast to score)

    Note: CatBoost uses 'loss_function' instead of sklearn's 'criterion'.
    Set verbose=0 to suppress the per-iteration console output.
    """
    if not CATBOOST_AVAILABLE:
        raise ImportError("catboost is required. Run: pip install catboost")

    return CatBoostClassifier(
        iterations      = trial.suggest_int  ("cb_clf_iterations",     100, 1000, log=True),
        depth           = trial.suggest_int  ("cb_clf_depth",             3,   10),
        learning_rate   = trial.suggest_float("cb_clf_lr",             0.01,  0.3, log=True),
        l2_leaf_reg     = trial.suggest_float("cb_clf_l2_leaf_reg",    1e-3, 10.0, log=True),
        bagging_temperature = trial.suggest_float("cb_clf_bagging_temp", 0.0,  1.0),
        random_strength = trial.suggest_float("cb_clf_random_strength", 1e-2, 10.0, log=True),
        border_count    = trial.suggest_int  ("cb_clf_border_count",     32,  254),
        random_seed     = 42,
        verbose         = 0,
        allow_writing_files = False,   # prevent CatBoost writing to disk during Optuna
    )


def catboost_regressor(trial):
    """CatBoost regressor — same hyperparameter space as the classifier."""
    if not CATBOOST_AVAILABLE:
        raise ImportError("catboost is required. Run: pip install catboost")

    return CatBoostRegressor(
        iterations      = trial.suggest_int  ("cb_reg_iterations",     100, 1000, log=True),
        depth           = trial.suggest_int  ("cb_reg_depth",             3,   10),
        learning_rate   = trial.suggest_float("cb_reg_lr",             0.01,  0.3, log=True),
        l2_leaf_reg     = trial.suggest_float("cb_reg_l2_leaf_reg",    1e-3, 10.0, log=True),
        bagging_temperature = trial.suggest_float("cb_reg_bagging_temp", 0.0,  1.0),
        random_strength = trial.suggest_float("cb_reg_random_strength", 1e-2, 10.0, log=True),
        border_count    = trial.suggest_int  ("cb_reg_border_count",     32,  254),
        random_seed     = 42,
        verbose         = 0,
        allow_writing_files = False,
    )
