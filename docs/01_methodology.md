# Methodology: The Synthetic Control Method (SCM)
## Mathematical Formulation and Identification Strategy

> **Project:** Causal Evaluation of Maharashtra EV Policy 2025  
> **Version:** 1.0 | **Date:** 2026-05-13

---

## 1. The Fundamental Problem of Causal Inference

For district `i` at time `t`, we observe only one potential outcome — treated `Y_it(1)` or untreated `Y_it(0)`. The individual treatment effect is:

```
τ_it = Y_it(1) - Y_it(0)
```

The **Average Treatment Effect (ATE)** over post-treatment period `T₀ < t ≤ T`:

```
ATE = (1/|T_post|) × Σ_{t=T₀+1}^{T} [Y_1t - Y_1t(0)]
```

The SCM constructs a data-driven estimate of the unobserved counterfactual `Y_1t(0)`.

---

## 2. SCM Estimator (Abadie et al., 2010)

### 2.1 Notation

- `J + 1` districts: district `1` = treated; districts `2..J+1` = donor pool
- `t = 1..T₀` pre-treatment (Jan 2019 – May 2025); `t = T₀+1..T` post-treatment (Jun 2025 – May 2026)
- `Y_it` = outcome (EV penetration rate or PM2.5 monthly mean)
- `X_i` = pre-treatment predictor vector for district `i`

### 2.2 Synthetic Control Construction

The synthetic control is a weighted combination of donor districts:

```
Ŷ_1t(0) = Σ_{j=2}^{J+1} w_j × Y_jt
```

Subject to: `w_j ≥ 0 ∀j` and `Σ w_j = 1` (non-negative, convex weights).

### 2.3 Weight Optimization (Nested)

**Outer problem** — choose predictor importance matrix `V*`:
```
V* = argmin_V  RMSPE_pre(W*(V))
RMSPE_pre = √( (1/T₀) × Σ_{t=1}^{T₀} (Y_1t - Ŷ_1t)² )
```

**Inner problem** — given `V`, choose district weights `W*`:
```
W*(V) = argmin_W  (X_1 - X_0·W)ᵀ V (X_1 - X_0·W)
s.t.  w_j ≥ 0,  Σ w_j = 1
```

Where `X_1` is (k×1) treated predictor vector, `X_0` is (k×J) donor predictor matrix.

### 2.4 Predictor Variables (X_i)

| Predictor | Definition | Rationale |
|-----------|------------|-----------|
| `ev_penetration_2019` | EV share of new registrations, 2019 | Baseline adoption |
| `ev_penetration_2022` | EV share, 2022 | Mid-period trend |
| `ev_penetration_2024` | EV share, 2024 | Near-policy baseline |
| `avg_ev_pen_pre` | Mean EV penetration 2019–2025 | Long-run rate |
| `gsdp_per_capita` | District GSDP per capita (₹ lakh) | Economic capacity |
| `urbanization_rate` | % urban population | Demand proxy |
| `charging_density_2024` | Stations per 100 km² | Pre-existing infra |
| `pm25_mean_2024` | Annual mean PM2.5 (μg/m³) | Air quality baseline |
| `two_wheeler_share` | 2W as % of all registrations | Fleet composition |

### 2.5 Treatment Effect Estimation

```
τ̂_1t = Y_1t - Ŷ_1t(0)        (per-period effect)
ATE  = (1/T_post) × Σ τ̂_1t   (average effect)
```

---

## 3. Inference: Placebo / Permutation Tests

### 3.1 In-Space Placebo (Cross-Sectional)

1. Iteratively assign "fake" treatment to each donor district `j`
2. Compute synthetic control and ATE for each placebo unit
3. Pseudo p-value: `p = #{j : |ATE_j| ≥ |ATE_1|} / (J+1)`
4. Exclude donors with pre-RMSPE > 5× treated pre-RMSPE

### 3.2 In-Time Placebo (Temporal)

Assign false treatment date within pre-period (e.g., Jan 2023). If model is valid, placebo effect ≈ 0.

### 3.3 RMSPE Ratio Statistic

```
R_j = RMSPE_post(j) / RMSPE_pre(j)
```

Large `R_1` relative to the permutation distribution of `R_j` indicates a significant effect.

---

## 4. Identification Assumptions

| Assumption | Description | Validation Strategy |
|------------|-------------|---------------------|
| **SUTVA** | No cross-district spillovers | Spatial buffer sensitivity analysis |
| **No hidden confounders** | No other differential policy shocks post-T₀ | FAME-II applies uniformly; DiD FE absorbs state shocks |
| **Pre-treatment fit** | Synthetic control replicates pre-treatment trajectory | RMSPE_pre < 0.15 pp threshold |
| **Convex hull** | Treated X_1 within convex hull of donor X_0 | Check no w*_j saturates at 1.0 |

---

## 5. Supplementary: Causal Forest (HTE via EconML)

Estimates **Conditional Average Treatment Effects (CATE)**:

```
CATE(x) = E[Y(1) - Y(0) | X = x]
```

Uses doubly-robust Robinson (1988) partial linear model:
1. Fit nuisance models: `μ̂_0(x) = E[Y|W=0,X]`, `ê(x) = P(W=1|X)`
2. Residualize: `Ỹ = Y - μ̂_0(X)`, `W̃ = W - ê(X)`
3. Forest on `Ỹ/W̃` → `CATE(x)`

```python
from econml.dml import CausalForestDML
model = CausalForestDML(n_estimators=1000, min_samples_leaf=10, random_state=42, cv=5)
model.fit(Y, W, X=X_controls)
```

---

## 6. Robustness: DiD with Two-Way Fixed Effects

```
Y_it = α_i + γ_t + β·(Treated_i × Post_t) + δ·X_it + ε_it
```

- `α_i` = district FE; `γ_t` = month-year FE; `β` = DiD ATE
- Standard errors clustered at district level
- Event-study specification validates parallel trends pre-T₀

---

## 7. Key References

1. Abadie, A. & Gardeazabal, J. (2003). *AER*, 93(1), 113–132.
2. Abadie, A., Diamond, A. & Hainmueller, J. (2010). *JASA*, 105(490), 493–505.
3. Wager, S. & Athey, S. (2018). *JASA*, 113(523), 1228–1242.
4. Holland, P. W. (1986). *JASA*, 81(396), 945–960.
5. Robinson, P. M. (1988). *Econometrica*, 56(4), 931–954.
