
import os
import time
import json
import numpy as np
import tensorflow as tf

from tensorflow.keras.applications.resnet50 import preprocess_input


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MODEL_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "app", "models", "multimodal_final.h5"),
    os.path.join(PROJECT_ROOT, "weights", "multimodal_final.h5"),
]

model_path = None
for path in MODEL_CANDIDATES:
    if os.path.exists(path):
        model_path = path
        break

if model_path is None:
    raise FileNotFoundError("Could not find multimodal_final.h5")

print("Loading model:", model_path)
model = tf.keras.models.load_model(model_path, compile=False)

N_WARMUP = 5
N_RUNS = 50

times = []

for i in range(N_WARMUP + N_RUNS):
    img_raw = np.random.rand(1, 224, 224, 3).astype(np.float32)
    img_pp = preprocess_input(img_raw * 255.0)

    tab = np.array([[25 / 30.0, 0.73, 0.0, 0.0]], dtype=np.float32)

    t0 = time.perf_counter()
    _ = model.predict([img_pp, tab], verbose=0)
    elapsed = time.perf_counter() - t0

    if i >= N_WARMUP:
        times.append(elapsed)

times = np.array(times)

result = {
    "runs": int(N_RUNS),
    "median_seconds": float(np.median(times)),
    "p95_seconds": float(np.percentile(times, 95)),
    "max_seconds": float(np.max(times)),
    "passed_p95_under_30s": bool(np.percentile(times, 95) < 30)
}

print(json.dumps(result, indent=2))

out_dir = os.path.join(PROJECT_ROOT, "reports", "day6")
os.makedirs(out_dir, exist_ok=True)

out_path = os.path.join(out_dir, "benchmark_performance.json")

with open(out_path, "w") as f:
    json.dump(result, f, indent=2)

print("Saved:", out_path)
print("PASS" if result["passed_p95_under_30s"] else "FAIL", "— spec requires p95 < 30s")
