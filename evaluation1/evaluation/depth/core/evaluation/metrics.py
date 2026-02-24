# from collections import OrderedDict

# import mmcv
# import numpy as np
# import torch


# def calculate(gt, pred):
#     if gt.shape[0] == 0:
#         return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

#     thresh = np.maximum((gt / pred), (pred / gt))
#     a1 = (thresh < 1.25).mean()
#     a2 = (thresh < 1.25**2).mean()
#     a3 = (thresh < 1.25**3).mean()

#     abs_rel = np.mean(np.abs(gt - pred) / gt)
#     sq_rel = np.mean(((gt - pred) ** 2) / gt)

#     rmse = (gt - pred) ** 2
#     rmse = np.sqrt(rmse.mean())

#     rmse_log = (np.log(gt) - np.log(pred)) ** 2
#     rmse_log = np.sqrt(rmse_log.mean())

#     err = np.log(pred) - np.log(gt)

#     silog = np.sqrt(np.mean(err**2) - np.mean(err) ** 2) * 100
#     if np.isnan(silog):
#         silog = 0

#     log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
#     return a1, a2, a3, abs_rel, rmse, log_10, rmse_log, silog, sq_rel


# def metrics(gt, pred, min_depth=1e-3, max_depth=80):
#     mask_1 = gt > min_depth
#     mask_2 = gt < max_depth
#     mask = np.logical_and(mask_1, mask_2)

#     gt = gt[mask]
#     pred = pred[mask]

#     a1, a2, a3, abs_rel, rmse, log_10, rmse_log, silog, sq_rel = calculate(gt, pred)

#     return a1, a2, a3, abs_rel, rmse, log_10, rmse_log, silog, sq_rel


# def eval_metrics(gt, pred, min_depth=1e-3, max_depth=80):
#     mask_1 = gt > min_depth
#     mask_2 = gt < max_depth
#     mask = np.logical_and(mask_1, mask_2)

#     gt = gt[mask]
#     pred = pred[mask]

#     thresh = np.maximum((gt / pred), (pred / gt))
#     a1 = (thresh < 1.25).mean()
#     a2 = (thresh < 1.25**2).mean()
#     a3 = (thresh < 1.25**3).mean()

#     abs_rel = np.mean(np.abs(gt - pred) / gt)
#     sq_rel = np.mean(((gt - pred) ** 2) / gt)

#     rmse = (gt - pred) ** 2
#     rmse = np.sqrt(rmse.mean())

#     rmse_log = (np.log(gt) - np.log(pred)) ** 2
#     rmse_log = np.sqrt(rmse_log.mean())

#     err = np.log(pred) - np.log(gt)
#     silog = np.sqrt(np.mean(err**2) - np.mean(err) ** 2) * 100

#     log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
#     return dict(
#         a1=a1,
#         a2=a2,
#         a3=a3,
#         abs_rel=abs_rel,
#         rmse=rmse,
#         log_10=log_10,
#         rmse_log=rmse_log,
#         silog=silog,
#         sq_rel=sq_rel,
#     )


# def pre_eval_to_metrics(pre_eval_results):

#     # convert list of tuples to tuple of lists, e.g.
#     # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
#     # ([A_1, ..., A_n], ..., [D_1, ..., D_n])
#     pre_eval_results = tuple(zip(*pre_eval_results))
#     ret_metrics = OrderedDict({})

#     ret_metrics["a1"] = np.nanmean(pre_eval_results[0])
#     ret_metrics["a2"] = np.nanmean(pre_eval_results[1])
#     ret_metrics["a3"] = np.nanmean(pre_eval_results[2])
#     ret_metrics["abs_rel"] = np.nanmean(pre_eval_results[3])
#     ret_metrics["rmse"] = np.nanmean(pre_eval_results[4])
#     ret_metrics["log_10"] = np.nanmean(pre_eval_results[5])
#     ret_metrics["rmse_log"] = np.nanmean(pre_eval_results[6])
#     ret_metrics["silog"] = np.nanmean(pre_eval_results[7])
#     ret_metrics["sq_rel"] = np.nanmean(pre_eval_results[8])

#     ret_metrics = {metric: value for metric, value in ret_metrics.items()}

#     return ret_metrics
from collections import OrderedDict

import mmcv
import numpy as np
import torch

def calculate(gt, pred):
    # Ensure float32 and flatten
    gt = np.asarray(gt, dtype=np.float32).reshape(-1)
    pred = np.asarray(pred, dtype=np.float32).reshape(-1)

    # Remove invalid / non‑positive entries (for division and log)
    mask = np.isfinite(gt) & np.isfinite(pred) & (gt > 0) & (pred > 0)
    if not np.any(mask):
        return (np.nan,) * 9

    gt = gt[mask]
    pred = pred[mask]

    if gt.shape[0] == 0:
        return (np.nan,) * 9

    # Threshold metrics
    ratio = gt / pred
    inv_ratio = pred / gt
    thresh = np.maximum(ratio, inv_ratio)

    a1 = np.mean(thresh < 1.25)
    a2 = np.mean(thresh < 1.25 * 1.25)
    a3 = np.mean(thresh < 1.25 * 1.25 * 1.25)

    # Errors
    diff = gt - pred
    diff2 = diff * diff

    abs_rel = np.mean(np.abs(diff) / gt)
    sq_rel = np.mean(diff2 / gt)

    rmse = np.sqrt(np.mean(diff2))

    # Logs
    log_gt = np.log(gt)
    log_pred = np.log(pred)

    log_diff = log_gt - log_pred
    log_diff2 = log_diff * log_diff
    rmse_log = np.sqrt(np.mean(log_diff2))

    err = log_pred - log_gt
    err2 = err * err
    silog = np.sqrt(np.mean(err2) - (np.mean(err) ** 2)) * 100.0
    if not np.isfinite(silog):
        silog = 0.0

    log_10 = np.mean(np.abs(np.log10(gt) - np.log10(pred)))

    return a1, a2, a3, abs_rel, rmse, log_10, rmse_log, silog, sq_rel


def metrics(gt, pred, min_depth=1e-3, max_depth=80):
    # Ensure float32 and flatten
    gt = np.asarray(gt, dtype=np.float32).reshape(-1)
    pred = np.asarray(pred, dtype=np.float32).reshape(-1)

    # Basic depth range mask
    mask = (gt > min_depth) & (gt < max_depth)
    if not np.any(mask):
        return (np.nan,) * 9

    gt = gt[mask]
    pred = pred[mask]

    return calculate(gt, pred)


def eval_metrics(gt_list, pred_list, min_depth=1e-3, max_depth=80):
    """Vectorized version that works on sequences of GT and Pred."""
    # gt_list and pred_list are iterables; we compute metrics per-pair
    a1s = []
    a2s = []
    a3s = []
    abs_rels = []
    rmses = []
    log_10s = []
    rmse_logs = []
    silogs = []
    sq_rels = []

    for gt, pred in zip(gt_list, pred_list):
        a1, a2, a3, abs_rel, rmse, log_10, rmse_log, silog, sq_rel = metrics(
            gt, pred, min_depth=min_depth, max_depth=max_depth
        )
        a1s.append(a1)
        a2s.append(a2)
        a3s.append(a3)
        abs_rels.append(abs_rel)
        rmses.append(rmse)
        log_10s.append(log_10)
        rmse_logs.append(rmse_log)
        silogs.append(silog)
        sq_rels.append(sq_rel)

    return dict(
        a1=a1s,
        a2=a2s,
        a3=a3s,
        abs_rel=abs_rels,
        rmse=rmses,
        log_10=log_10s,
        rmse_log=rmse_logs,
        silog=silogs,
        sq_rel=sq_rels,
    )


def pre_eval_to_metrics(pre_eval_results):
    # convert list of tuples to tuple of lists, e.g.
    # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
    # ([A_1, ..., A_n], ..., [D_1, ..., D_n])
    pre_eval_results = tuple(zip(*pre_eval_results))
    ret_metrics = OrderedDict({})

    ret_metrics["a1"] = np.nanmean(pre_eval_results[0])
    ret_metrics["a2"] = np.nanmean(pre_eval_results[1])
    ret_metrics["a3"] = np.nanmean(pre_eval_results[2])
    ret_metrics["abs_rel"] = np.nanmean(pre_eval_results[3])
    ret_metrics["rmse"] = np.nanmean(pre_eval_results[4])
    ret_metrics["log_10"] = np.nanmean(pre_eval_results[5])
    ret_metrics["rmse_log"] = np.nanmean(pre_eval_results[6])
    ret_metrics["silog"] = np.nanmean(pre_eval_results[7])
    ret_metrics["sq_rel"] = np.nanmean(pre_eval_results[8])

    ret_metrics = {metric: value for metric, value in ret_metrics.items()}

    return ret_metrics