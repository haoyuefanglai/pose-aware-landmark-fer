"""
frame_recorder.py
逐帧记录与视频级汇总。

补齐的缺口：原实现里 realtime_demo.py 只支持按 [S] 存截图，不落任何"第几帧、
什么姿态、预测成什么、分数多少"的逐帧记录；pose_stability_test.py 的输入又写死
在 FER2013 静态图上、读不了视频文件。于是任务C 的"视频链路"能演示，却产不出
可写进报告的数字。

本模块把"逐帧落盘 + 视频级汇总"从具体脚本里抽出来，供
  - src/experiments/evaluate_video.py （批量评测视频文件/摄像头）
  - src/apps/realtime_demo.py          （交互演示时可选 --record-csv）
共用，保证两条路径产出的 CSV 与汇总口径完全一致。
"""

# --- 路径引导：src/ 下 core / pipeline / experiments / apps 之间可互相 import ---
import os as _os
import sys as _sys

_SRC_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _sub in ("core", "pipeline", "experiments", "apps"):
    _sub_path = _os.path.join(_SRC_DIR, _sub)
    if _os.path.isdir(_sub_path) and _sub_path not in _sys.path:
        _sys.path.insert(0, _sub_path)
# --- 路径引导结束 ---

import os

import numpy as np

# 逐帧记录的列定义（顺序即 CSV 列序）
FRAME_COLUMNS = [
    "frame", "time_s", "detected", "latency_ms",
    "yaw", "pitch", "roll", "vis_mean", "vis_min",
    "pred_raw", "pred_smooth", "score", "rejected", "engine", "calibrated",
]

# 头部姿态分桶（与 pose_stability_test.py 保持一致，便于两处数字并列比较）
POSE_BINS = {
    "yaw": ([0, 5, 15, 25, np.inf], ["|yaw|<=5", "5<|yaw|<=15", "15<|yaw|<=25", "|yaw|>25"]),
    "pitch": ([0, 10, 20, 35, np.inf], ["|pitch|<=10", "10<|pitch|<=20", "20<|pitch|<=35", "|pitch|>35"]),
    "roll": ([0, 10, 20, 35, np.inf], ["|roll|<=10", "10<|roll|<=20", "20<|roll|<=35", "|roll|>35"]),
}


def _bin_of(values, edges, names):
    """按 |value| 落入的区间返回桶名，超出最后一个边界归入最后一桶。"""
    a = np.abs(np.asarray(values, dtype=np.float64))
    out = np.full(len(a), "n/a", dtype=object)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        out[(a > lo) & (a <= hi)] = names[i]
    out[a <= edges[0]] = names[0]
    out[np.isnan(a)] = "n/a"
    return out


class FrameRecorder:
    """收集逐帧记录，并产出一致的视频级汇总统计。"""

    def __init__(self, engine="ml", calibrated=False, source=""):
        self.engine = engine
        self.calibrated = bool(calibrated)
        self.source = str(source)
        self.rows = []

    # ---------------------------------------------------------------- 写入
    def add(self, frame_idx, time_s, detected, latency_ms,
            yaw=None, pitch=None, roll=None, vis_mean=None, vis_min=None,
            pred_raw=None, pred_smooth=None, score=None, rejected=False):
        """写入一帧。未检出时给出 detected=False 即可，其余字段留空。"""
        self.rows.append({
            "frame": int(frame_idx),
            "time_s": round(float(time_s), 4),
            "detected": bool(detected),
            "latency_ms": round(float(latency_ms), 3),
            "yaw": None if yaw is None else round(float(yaw), 3),
            "pitch": None if pitch is None else round(float(pitch), 3),
            "roll": None if roll is None else round(float(roll), 3),
            "vis_mean": None if vis_mean is None else round(float(vis_mean), 4),
            "vis_min": None if vis_min is None else round(float(vis_min), 4),
            "pred_raw": pred_raw,
            "pred_smooth": pred_smooth,
            "score": None if score is None else round(float(score), 4),
            "rejected": bool(rejected),
            "engine": self.engine,
            "calibrated": self.calibrated,
        })

    # ---------------------------------------------------------------- 输出
    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame(self.rows, columns=FRAME_COLUMNS)

    def save_csv(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        df = self.to_dataframe()
        df.to_csv(path, index=False)
        return path

    # ---------------------------------------------------------------- 汇总
    def summary(self, classes=None):
        """视频级汇总。数字全部来自实际跑过的帧，可直接写进实验报告。"""
        import pandas as pd
        df = self.to_dataframe()
        n = len(df)
        if n == 0:
            return {"frames": 0}

        det = df[df["detected"]].copy()
        out = {
            "source": self.source,
            "engine": self.engine,
            "confidence_kind": "calibrated_probability" if self.calibrated else "uncalibrated_relative_score",
            "frames": int(n),
            "detected_frames": int(len(det)),
            "detect_rate": round(float(len(det) / n), 4),
            "latency_ms_mean": round(float(df["latency_ms"].mean()), 2),
            "latency_ms_p50": round(float(df["latency_ms"].median()), 2),
            "latency_ms_p95": round(float(np.percentile(df["latency_ms"], 95)), 2),
            "throughput_fps": round(float(1000.0 / max(df["latency_ms"].mean(), 1e-9)), 2),
        }

        if len(det) == 0:
            out["note"] = "整段视频未检出人脸"
            return out

        # ---- 姿态分桶：按已检出帧的姿态角分桶，统计可见性 / 分数 / 标签一致性 ----
        # 未检出的帧拿不到姿态角，因此单独作为一行报告，不混进任何姿态桶里。
        pose = {}
        for key in ("yaw", "pitch", "roll"):
            edges, names = POSE_BINS[key]
            bins = _bin_of(det[key].to_numpy(dtype=np.float64), edges, names)
            rows = []
            for name in names:
                sub = det[bins == name]
                row = {
                    "bin": name,
                    "n": int(len(sub)),
                    "share": round(float(len(sub) / len(det)), 4) if len(det) else None,
                    "vis_mean": round(float(sub["vis_mean"].mean()), 4) if len(sub) else None,
                    "score_mean": round(float(sub["score"].mean()), 4) if len(sub) else None,
                    "top_label": None,
                    "top_label_share": None,
                }
                if len(sub):
                    vc = sub["pred_smooth"].value_counts(normalize=True)
                    row["top_label"] = str(vc.index[0])
                    row["top_label_share"] = round(float(vc.iloc[0]), 4)
                rows.append(row)
            rows.append({
                "bin": "未检出(无姿态角)", "n": int(n - len(det)),
                "share": round(float((n - len(det)) / n), 4),
                "vis_mean": None, "score_mean": None,
                "top_label": None, "top_label_share": None,
            })
            pose[key] = rows
        out["pose_bins"] = pose

        # ---- 可见性与姿态的关系（证明可见性列真的随姿态变化）----
        y = np.abs(df["yaw"].to_numpy(dtype=np.float64))
        v = df["vis_mean"].to_numpy(dtype=np.float64)
        fin = np.isfinite(y) & np.isfinite(v)
        out["visibility"] = {
            "mean": round(float(np.nanmean(v[fin])), 4) if fin.any() else None,
            "min": round(float(np.nanmin(v[fin])), 4) if fin.any() else None,
            "corr_abs_yaw": (round(float(np.corrcoef(y[fin], v[fin])[0, 1]), 4)
                             if fin.sum() > 2 and np.std(v[fin]) > 0 else None),
        }

        # ---- 时序稳定性：标签切换次数与平均连续段长 ----
        det_sorted = det.sort_values("frame")
        labels = det_sorted["pred_smooth"].fillna("None").to_numpy()
        switches = int(np.sum(labels[1:] != labels[:-1])) if len(labels) > 1 else 0
        runs = []
        cur, cnt = None, 0
        for lab in labels:
            if lab == cur:
                cnt += 1
            else:
                if cur is not None:
                    runs.append(cnt)
                cur, cnt = lab, 1
        if cur is not None:
            runs.append(cnt)
        out["temporal"] = {
            "label_switches": switches,
            "switches_per_100_frames": round(100.0 * switches / max(len(labels) - 1, 1), 2),
            "mean_run_length": round(float(np.mean(runs)), 2) if runs else None,
            "max_run_length": int(max(runs)) if runs else None,
        }

        # ---- 分数分布与拒识比例 ----
        scores = det["score"].to_numpy(dtype=np.float64)
        scores = scores[np.isfinite(scores)]
        out["scores"] = {
            "mean": round(float(scores.mean()), 4) if len(scores) else None,
            "p50": round(float(np.median(scores)), 4) if len(scores) else None,
            "p10": round(float(np.percentile(scores, 10)), 4) if len(scores) else None,
            "rejected_frames": int(det["rejected"].sum()),
            "rejected_rate": round(float(det["rejected"].mean()), 4),
        }

        # ---- 标签占比（平滑后的稳定输出）----
        occ = det_sorted["pred_smooth"].fillna("None").value_counts(normalize=True)
        out["label_occupancy"] = {str(k): round(float(v), 4) for k, v in occ.items()}
        if classes:
            out["label_occupancy"] = {c: out["label_occupancy"].get(c, 0.0) for c in classes
                                      if c in out["label_occupancy"]} or out["label_occupancy"]
        return out

    def summary_text(self, summary=None):
        """把汇总渲染成人类可读的多行文本（供控制台/报告引用）。"""
        s = summary or self.summary()
        if not s.get("frames"):
            return "（无帧）"
        L = []
        L.append(f"视频源: {s['source']}   引擎: {s['engine']}   置信度性质: {s['confidence_kind']}")
        L.append(f"帧数: {s['frames']}   检出: {s['detected_frames']} ({s['detect_rate']*100:.1f}%)")
        L.append(f"单帧耗时: 均值 {s['latency_ms_mean']:.1f}ms / P50 {s['latency_ms_p50']:.1f}ms / "
                 f"P95 {s['latency_ms_p95']:.1f}ms  → 约 {s['throughput_fps']:.1f} FPS")
        vis = s.get("visibility") or {}
        if vis.get("mean") is not None:
            L.append(f"可见性: 均值 {vis['mean']:.3f} / 最低 {vis['min']:.3f} / "
                     f"corr(|yaw|, vis) = {vis['corr_abs_yaw']}")
        if "pose_bins" in s:
            L.append("姿态分桶（yaw，仅统计已检出帧）:")
            for r in s["pose_bins"]["yaw"]:
                L.append("  %-18s n=%-5d 占比=%-6s vis=%-7s score=%-7s 主标签=%s(%.0f%%)" % (
                    r["bin"], r["n"],
                    "-" if r["share"] is None else f"{r['share']*100:.1f}%",
                    "-" if r["vis_mean"] is None else f"{r['vis_mean']:.3f}",
                    "-" if r["score_mean"] is None else f"{r['score_mean']:.3f}",
                    r.get("top_label") or "-",
                    (r.get("top_label_share") or 0) * 100))
        t = s.get("temporal") or {}
        if t:
            L.append(f"时序稳定: 标签切换 {t['label_switches']} 次 "
                     f"({t['switches_per_100_frames']:.1f}/100帧)，平均连续段长 {t['mean_run_length']} 帧")
        sc = s.get("scores") or {}
        if sc:
            L.append(f"分数: 均值 {sc['mean']} / P50 {sc['p50']} / P10 {sc['p10']}，"
                     f"拒识 {sc['rejected_frames']} 帧 ({sc['rejected_rate']*100:.1f}%)")
        occ = s.get("label_occupancy") or {}
        if occ:
            L.append("标签占比: " + "  ".join(f"{k}={v*100:.1f}%" for k, v in occ.items()))
        return "\n".join(L)
