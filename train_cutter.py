"""
train_cutter.py — Train auto-cut model from splits.json + PDF

Usage:
    python train_cutter.py

Reads:
    splits.json     — your 50 labeled questions (b1, r1, r2, b2 per page)
    PDF file        — renders each page to extract pixel features

Outputs:
    train_cutter.pkl   — trained model (4 regressors + metadata)
    train_report.txt   — accuracy per line, worst predictions

Once trained, splitter.py loads this pkl and auto-places all 4 lines
when you open a new page. You only drag to correct mistakes.
"""

import json, pickle, sys, os
import numpy as np

# ── Config ─────────────────────────────────────────────────────────
PDF_PATH    = r"D:\Study meteial\NEET PG QBANK\pyq\5.Micro PYQ (2017-2022).pdf"
SPLITS_JSON = r"D:\Study meteial\NEET PG QBANK\p2e\output\5.Micro PYQ (2017-2022)\splits.json"
MODEL_OUT   = r"D:\Study meteial\NEET PG QBANK\p2e\train_cutter.pkl"
REPORT_OUT  = r"D:\Study meteial\NEET PG QBANK\p2e\train_report.txt"

RENDER_DPI  = 72    # lower = faster, enough for brightness profile
# ───────────────────────────────────────────────────────────────────


def render_page_gray(doc, page_idx: int, dpi: int = 72) -> np.ndarray:
    """Render a PDF page → grayscale numpy array (H,)  — brightness profile."""
    import pypdfium2 as pdfium
    page   = doc[page_idx]
    scale  = dpi / 72.0
    bitmap = page.render(scale=scale, rotation=0)
    pil    = bitmap.to_pil()
    gray   = pil.convert("L")
    arr    = np.array(gray, dtype=np.float32)          # (H, W)
    # Row brightness profile: mean pixel per row, normalized 0-1
    profile = arr.mean(axis=1) / 255.0                  # (H,)
    return profile


def extract_features(profile: np.ndarray, n_bins: int = 100) -> np.ndarray:
    """
    Convert variable-length brightness profile → fixed-length feature vector.

    Features:
    - 100-bin histogram of row brightness (captures text density distribution)
    - Mean brightness of 10 equal vertical zones (captures top/middle/bottom layout)
    - Gradient of brightness profile (captures transitions between regions)
    - 5th, 25th, 50th, 75th, 95th percentile brightness
    """
    h = len(profile)

    # 1. Histogram (100 bins)
    hist, _ = np.histogram(profile, bins=n_bins, range=(0.0, 1.0))
    hist    = hist.astype(np.float32) / (h + 1e-9)

    # 2. Zonal means (10 zones)
    zones = np.array_split(profile, 10)
    zone_means = np.array([z.mean() for z in zones], dtype=np.float32)

    # 3. Gradient — detect sharp transitions (question → options boundary etc.)
    grad = np.abs(np.gradient(profile)).astype(np.float32)
    # Downsample gradient to 50 points
    grad_ds = np.interp(
        np.linspace(0, len(grad)-1, 50),
        np.arange(len(grad)),
        grad
    ).astype(np.float32)

    # 4. Percentiles
    pcts = np.percentile(profile, [5, 25, 50, 75, 95]).astype(np.float32)

    # 5. Top-10 gradient peaks positions (where lines likely are)
    peak_idx = np.argsort(grad)[-10:][::-1]
    peak_pos = np.sort(peak_idx / (h + 1e-9)).astype(np.float32)  # normalized

    feat = np.concatenate([hist, zone_means, grad_ds, pcts, peak_pos])
    return feat


def main():
    # ── Load labels ────────────────────────────────────────────────
    print("Loading splits.json …")
    with open(SPLITS_JSON, encoding="utf-8") as f:
        splits = json.load(f)

    labels = []   # (page_idx, b1, r1, r2, b2)
    for v in splits.values():
        labels.append((
            int(v["page_idx"]),
            float(v["b1"]),
            float(v["r1"]),
            float(v["r2"]),
            float(v["b2"]),
        ))
    print(f"  {len(labels)} labeled pages")

    # ── Open PDF ───────────────────────────────────────────────────
    print(f"Opening PDF: {PDF_PATH}")
    try:
        import pypdfium2 as pdfium
    except ImportError:
        print("ERROR: pypdfium2 not installed — run: pip install pypdfium2")
        sys.exit(1)

    doc = pdfium.PdfDocument(PDF_PATH)
    print(f"  PDF has {len(doc)} pages")

    # ── Extract features ───────────────────────────────────────────
    print("Rendering pages and extracting features …")
    X, Y = [], []
    failed = 0
    for i, (page_idx, b1, r1, r2, b2) in enumerate(labels):
        try:
            profile = render_page_gray(doc, page_idx, RENDER_DPI)
            feat    = extract_features(profile)
            X.append(feat)
            Y.append([b1, r1, r2, b2])
            if (i+1) % 10 == 0:
                print(f"  {i+1}/{len(labels)} pages processed")
        except Exception as e:
            print(f"  WARNING: page {page_idx} failed: {e}")
            failed += 1

    print(f"  Done. {len(X)} features extracted, {failed} failed")
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)

    # ── Train ──────────────────────────────────────────────────────
    print("\nTraining models …")
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import cross_val_score
    except ImportError:
        print("ERROR: scikit-learn not installed — run: pip install scikit-learn")
        sys.exit(1)

    LINE_NAMES = ["b1 (blue/top)", "r1 (red/stem)", "r2 (orange/opts)", "b2 (purple/exp)"]
    models  = []
    report  = ["AUTO-CUT MODEL TRAINING REPORT", "="*50, ""]

    for i, name in enumerate(LINE_NAMES):
        y = Y[:, i]
        model = GradientBoostingRegressor(
            n_estimators  = 200,
            learning_rate = 0.05,
            max_depth     = 4,
            subsample     = 0.8,
            random_state  = 42,
        )
        # Cross-validation MAE (mean absolute error in fraction units)
        cv_scores = cross_val_score(
            model, X, y,
            cv      = min(5, len(X)//3),
            scoring = "neg_mean_absolute_error"
        )
        mae_cv = -cv_scores.mean()

        # Fit on all data
        model.fit(X, y)
        train_preds = model.predict(X)
        mae_train   = np.abs(train_preds - y).mean()

        # Worst predictions
        errors  = np.abs(train_preds - y)
        worst_i = np.argsort(errors)[-3:][::-1]

        print(f"  {name:<25} CV-MAE={mae_cv:.4f}  train-MAE={mae_train:.4f}")
        models.append(model)

        report.append(f"LINE: {name}")
        report.append(f"  CV MAE   : {mae_cv:.4f}  ({mae_cv*100:.2f}% of page height)")
        report.append(f"  Train MAE: {mae_train:.4f}")
        report.append(f"  Range    : {y.min():.3f} – {y.max():.3f}")
        report.append(f"  Worst predictions:")
        for wi in worst_i:
            report.append(f"    page {labels[wi][0]}: pred={train_preds[wi]:.3f}  actual={y[wi]:.3f}  err={errors[wi]:.3f}")
        report.append("")

    # ── Save model ─────────────────────────────────────────────────
    payload = {
        "models"    : models,
        "n_features": X.shape[1],
        "render_dpi": RENDER_DPI,
        "line_names": LINE_NAMES,
        "n_train"   : len(X),
    }
    with open(MODEL_OUT, "wb") as f:
        pickle.dump(payload, f)
    print(f"\nModel saved → {MODEL_OUT}")

    # ── Save report ────────────────────────────────────────────────
    report.append(f"Total training samples: {len(X)}")
    report.append(f"Feature vector size   : {X.shape[1]}")
    report.append(f"Render DPI            : {RENDER_DPI}")
    with open(REPORT_OUT, "w") as f:
        f.write("\n".join(report))
    print(f"Report saved → {REPORT_OUT}")

    # ── Quick summary ───────────────────────────────────────────────
    print("\n── SUMMARY ──────────────────────────────────────────")
    print("MAE is fraction of page height. 0.02 = 2% off = ~20px on a typical page.")
    print("For splitter use, anything < 0.04 is good (lines snap close enough to drag).")
    print("\nNext step: copy train_cutter.pkl to your p2e/ folder.")
    print("Then re-run splitter.py — it will auto-detect the pkl and pre-place lines.")


if __name__ == "__main__":
    main()
