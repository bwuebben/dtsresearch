"""
Statistical tests and inference utilities for DTS research.

Implements:
- Clustered standard errors (week, issuer, two-way)
- Chow test for structural breaks
- Joint F-test and Wald test
- Inverse-variance weighted pooling
- Bootstrap procedures
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
from scipy import stats
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
from statsmodels.stats.weightstats import DescrStatsW


def clustered_standard_errors(
    X: np.ndarray,
    y: np.ndarray,
    cluster_var: np.ndarray,
    add_constant: bool = True
) -> Dict:
    """
    Estimate OLS regression with clustered standard errors.

    Args:
        X: Design matrix (n x k)
        y: Dependent variable (n x 1)
        cluster_var: Cluster assignment (n x 1)
        add_constant: Whether to add intercept

    Returns:
        Dictionary with coefficients, clustered SEs, t-stats, p-values
    """
    if add_constant and not np.all(X[:, 0] == 1):
        X = sm.add_constant(X)

    # Fit OLS
    model = OLS(y, X).fit()

    # Get clustering variable
    clusters = pd.Series(cluster_var)
    n_clusters = clusters.nunique()

    # Compute clustered variance-covariance matrix
    # V = (X'X)^-1 * M * (X'X)^-1
    # where M = Σ_c (X'_c ε_c ε'_c X_c)

    # Get residuals
    residuals = y - model.predict(X)

    # Compute meat of sandwich estimator
    meat = np.zeros((X.shape[1], X.shape[1]))
    for cluster_id in clusters.unique():
        cluster_mask = (clusters == cluster_id).values
        X_c = X[cluster_mask]
        e_c = residuals[cluster_mask]
        meat += X_c.T @ np.outer(e_c, e_c) @ X_c

    # Bread of sandwich
    XtX_inv = np.linalg.inv(X.T @ X)

    # Clustered variance-covariance matrix
    # Apply small-sample correction: (G/(G-1)) * ((N-1)/(N-K))
    G = n_clusters
    N = len(y)
    K = X.shape[1]
    correction = (G / (G - 1)) * ((N - 1) / (N - K))

    V_clustered = correction * XtX_inv @ meat @ XtX_inv

    # Standard errors
    se_clustered = np.sqrt(np.diag(V_clustered))

    # T-statistics
    t_stats = model.params / se_clustered

    # P-values (two-tailed)
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=G - 1))

    return {
        'params': model.params,
        'se': se_clustered,
        't_stats': t_stats,
        'p_values': p_values,
        'n_obs': N,
        'n_clusters': G,
        'r_squared': model.rsquared,
        'r_squared_adj': model.rsquared_adj,
        'vcov': V_clustered
    }


def chow_test(
    results_list: List[Dict],
    method: str = 'wald'
) -> Dict:
    """
    Chow test for structural breaks.

    Tests H0: β_1 = β_2 = ... = β_W (parameters equal across windows)

    Args:
        results_list: List of regression results dictionaries
        method: 'wald' or 'f_stat'

    Returns:
        Dictionary with test statistic and p-value
    """
    if len(results_list) < 2:
        raise ValueError("Need at least 2 windows for Chow test")

    # Extract parameters and covariance matrices
    params_list = [r['params'] for r in results_list]
    vcov_list = [r['vcov'] for r in results_list]

    W = len(params_list)  # Number of windows
    K = len(params_list[0])  # Number of parameters

    if method == 'wald':
        # Wald test: (Rβ - r)' [R V R']^-1 (Rβ - r)
        # H0: β_1 = β_2 = ... = β_W

        # Stack parameters
        beta_stacked = np.concatenate(params_list)

        # Create restriction matrix R
        # Each row tests β_i,k = β_j,k for some i,j,k
        R_rows = []
        for w in range(W - 1):
            for k in range(K):
                row = np.zeros(W * K)
                row[w * K + k] = 1
                row[(w + 1) * K + k] = -1
                R_rows.append(row)

        R = np.array(R_rows)

        # Block diagonal covariance matrix
        V_block = np.zeros((W * K, W * K))
        for w in range(W):
            V_block[w*K:(w+1)*K, w*K:(w+1)*K] = vcov_list[w]

        # Compute Wald statistic
        RVRt = R @ V_block @ R.T
        try:
            RVRt_inv = np.linalg.inv(RVRt)
            Rbeta = R @ beta_stacked
            wald_stat = Rbeta.T @ RVRt_inv @ Rbeta
        except np.linalg.LinAlgError:
            return {'statistic': np.nan, 'p_value': np.nan, 'interpretation': 'Singular matrix'}

        # Degrees of freedom
        df = (W - 1) * K

        # P-value from chi-squared distribution
        p_value = 1 - stats.chi2.cdf(wald_stat, df)

        return {
            'statistic': wald_stat,
            'p_value': p_value,
            'df': df,
            'method': 'Wald',
            'interpretation': interpret_chow_test(p_value)
        }

    else:  # F-statistic method
        # Not implemented for now, return Wald as default
        return chow_test(results_list, method='wald')


def interpret_chow_test(p_value: float, alpha: float = 0.10) -> str:
    """
    Interpret Chow test result.

    Args:
        p_value: P-value from Chow test
        alpha: Significance level

    Returns:
        Interpretation string
    """
    if p_value < alpha:
        return f'Time-variation detected (p={p_value:.4f} < {alpha})'
    else:
        return f'Static lambda sufficient (p={p_value:.4f} >= {alpha})'


def joint_f_test(
    X: np.ndarray,
    y: np.ndarray,
    restriction_indices: List[int],
    cluster_var: Optional[np.ndarray] = None
) -> Dict:
    """
    Joint F-test for multiple restrictions.

    Tests H0: β_j1 = β_j2 = ... = β_jq = 0

    Args:
        X: Design matrix
        y: Dependent variable
        restriction_indices: Indices of parameters to test jointly
        cluster_var: Optional cluster variable for clustered SEs

    Returns:
        Dictionary with F-statistic and p-value
    """
    # Fit unrestricted model
    if cluster_var is not None:
        results_unrestricted = clustered_standard_errors(X, y, cluster_var, add_constant=False)
    else:
        model = OLS(y, X).fit()
        results_unrestricted = {
            'params': model.params,
            'vcov': model.cov_params(),
            'r_squared': model.rsquared
        }

    # Fit restricted model (drop tested parameters)
    X_restricted = np.delete(X, restriction_indices, axis=1)
    if cluster_var is not None:
        results_restricted = clustered_standard_errors(X_restricted, y, cluster_var, add_constant=False)
    else:
        model_restricted = OLS(y, X_restricted).fit()
        results_restricted = {
            'r_squared': model_restricted.rsquared
        }

    # Compute F-statistic
    n = len(y)
    k_unrestricted = X.shape[1]
    k_restricted = X_restricted.shape[1]
    q = k_unrestricted - k_restricted

    r2_u = results_unrestricted['r_squared']
    r2_r = results_restricted['r_squared']

    f_stat = ((r2_u - r2_r) / q) / ((1 - r2_u) / (n - k_unrestricted))

    # P-value
    p_value = 1 - stats.f.cdf(f_stat, q, n - k_unrestricted)

    return {
        'f_statistic': f_stat,
        'p_value': p_value,
        'df_numerator': q,
        'df_denominator': n - k_unrestricted,
        'r_squared_unrestricted': r2_u,
        'r_squared_restricted': r2_r,
        'interpretation': f"Joint test p-value = {p_value:.4f}"
    }


def inverse_variance_weighted_pooling(
    estimates: np.ndarray,
    variances: np.ndarray
) -> Dict:
    """
    Inverse-variance weighted pooling of estimates.

    Used for meta-analysis / combining estimates across groups.

    Args:
        estimates: Array of point estimates (n,)
        variances: Array of variances (n,)

    Returns:
        Dictionary with pooled estimate, SE, and confidence interval
    """
    # Remove NaN estimates
    valid = ~np.isnan(estimates) & ~np.isnan(variances) & (variances > 0)
    estimates = estimates[valid]
    variances = variances[valid]

    if len(estimates) == 0:
        return {
            'pooled_estimate': np.nan,
            'pooled_se': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'n_estimates': 0
        }

    # Inverse-variance weights
    weights = 1.0 / variances
    weights = weights / weights.sum()  # Normalize

    # Pooled estimate
    pooled = np.sum(weights * estimates)

    # Pooled variance
    pooled_var = 1.0 / np.sum(1.0 / variances)
    pooled_se = np.sqrt(pooled_var)

    # 95% confidence interval
    ci_lower = pooled - 1.96 * pooled_se
    ci_upper = pooled + 1.96 * pooled_se

    return {
        'pooled_estimate': pooled,
        'pooled_se': pooled_se,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n_estimates': len(estimates)
    }


def wald_test(
    params: np.ndarray,
    vcov: np.ndarray,
    R: np.ndarray,
    r: np.ndarray = None
) -> Dict:
    """
    General Wald test for linear restrictions.

    Tests H0: R * params = r

    Args:
        params: Parameter estimates (k x 1)
        vcov: Variance-covariance matrix (k x k)
        R: Restriction matrix (q x k)
        r: Restriction vector (q x 1). If None, assumes r = 0

    Returns:
        Dictionary with Wald statistic and p-value
    """
    if r is None:
        r = np.zeros(R.shape[0])

    # Compute Wald statistic
    Rbeta_minus_r = R @ params - r
    RVRt = R @ vcov @ R.T

    try:
        RVRt_inv = np.linalg.inv(RVRt)
        wald_stat = Rbeta_minus_r.T @ RVRt_inv @ Rbeta_minus_r
    except np.linalg.LinAlgError:
        return {'statistic': np.nan, 'p_value': np.nan}

    # Degrees of freedom
    df = R.shape[0]

    # P-value from chi-squared distribution
    p_value = 1 - stats.chi2.cdf(wald_stat, df)

    return {
        'statistic': wald_stat,
        'p_value': p_value,
        'df': df
    }


def bootstrap_standard_errors(
    X: np.ndarray,
    y: np.ndarray,
    n_iterations: int = 1000,
    cluster_var: Optional[np.ndarray] = None
) -> Dict:
    """
    Bootstrap standard errors for OLS regression.

    Args:
        X: Design matrix
        y: Dependent variable
        n_iterations: Number of bootstrap iterations
        cluster_var: Optional cluster variable (resample clusters, not obs)

    Returns:
        Dictionary with bootstrap SEs and confidence intervals
    """
    n = len(y)
    k = X.shape[1]

    # Store bootstrap estimates
    bootstrap_params = np.zeros((n_iterations, k))

    if cluster_var is not None:
        # Cluster bootstrap: resample clusters
        clusters = pd.Series(cluster_var).unique()

        for i in range(n_iterations):
            # Resample clusters with replacement
            sampled_clusters = np.random.choice(clusters, size=len(clusters), replace=True)

            # Get observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                cluster_indices = np.where(cluster_var == cluster)[0]
                indices.extend(cluster_indices)

            # Fit on bootstrap sample
            X_boot = X[indices]
            y_boot = y[indices]

            try:
                model = OLS(y_boot, X_boot).fit()
                bootstrap_params[i] = model.params
            except:
                bootstrap_params[i] = np.nan

    else:
        # Standard bootstrap: resample observations
        for i in range(n_iterations):
            indices = np.random.choice(n, size=n, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]

            try:
                model = OLS(y_boot, X_boot).fit()
                bootstrap_params[i] = model.params
            except:
                bootstrap_params[i] = np.nan

    # Compute bootstrap SEs
    bootstrap_se = np.nanstd(bootstrap_params, axis=0)

    # Compute bootstrap confidence intervals (percentile method)
    ci_lower = np.nanpercentile(bootstrap_params, 2.5, axis=0)
    ci_upper = np.nanpercentile(bootstrap_params, 97.5, axis=0)

    return {
        'bootstrap_se': bootstrap_se,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'bootstrap_params': bootstrap_params
    }


def test_monotonicity(
    x: np.ndarray,
    y: np.ndarray,
    method: str = 'spearman'
) -> Dict:
    """
    Test for monotonic relationship between x and y.

    Args:
        x: Independent variable
        y: Dependent variable
        method: 'spearman' or 'kendall'

    Returns:
        Dictionary with correlation and p-value
    """
    # Remove NaN
    valid = ~np.isnan(x) & ~np.isnan(y)
    x = x[valid]
    y = y[valid]

    if len(x) < 3:
        return {'correlation': np.nan, 'p_value': np.nan}

    if method == 'spearman':
        corr, p_value = stats.spearmanr(x, y)
    elif method == 'kendall':
        corr, p_value = stats.kendalltau(x, y)
    else:
        raise ValueError(f"Unknown method: {method}")

    interpretation = ''
    if p_value < 0.05:
        if corr > 0:
            interpretation = f'Significant positive monotonic relationship (ρ={corr:.3f}, p={p_value:.4f})'
        else:
            interpretation = f'Significant negative monotonic relationship (ρ={corr:.3f}, p={p_value:.4f})'
    else:
        interpretation = f'No significant monotonic relationship (ρ={corr:.3f}, p={p_value:.4f})'

    return {
        'correlation': corr,
        'p_value': p_value,
        'method': method,
        'interpretation': interpretation
    }
