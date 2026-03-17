"""
Load Test Script — Wine Quality Prediction API
Sends concurrent prediction requests for monitoring/testing.
"""

import argparse
import json
import random
import time
import threading
import urllib.request
import urllib.error
from collections import Counter

# Sample wine data ranges (based on real dataset statistics)
FEATURE_RANGES = {
    "fixed acidity":       (3.8, 15.9),
    "volatile acidity":    (0.08, 1.58),
    "citric acid":         (0.0, 1.66),
    "residual sugar":      (0.6, 65.8),
    "chlorides":           (0.009, 0.611),
    "free sulfur dioxide": (1.0, 289.0),
    "total sulfur dioxide":(6.0, 440.0),
    "density":             (0.987, 1.039),
    "pH":                  (2.72, 4.01),
    "sulphates":           (0.22, 2.0),
    "alcohol":             (8.0, 14.9),
    "wine_type":           (0, 1),
}


def generate_sample():
    """Generate a random wine sample within realistic ranges."""
    sample = {}
    for feat, (lo, hi) in FEATURE_RANGES.items():
        if feat == "wine_type":
            sample[feat] = random.choice([0, 1])
        else:
            sample[feat] = round(random.uniform(lo, hi), 4)
    return sample


def send_request(url, sample, results, idx):
    """Send a single prediction request."""
    start = time.time()
    data = json.dumps(sample).encode("utf-8")
    req = urllib.request.Request(
        f"{url}/predict",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read().decode())
            duration = time.time() - start
            results[idx] = {
                "status": resp.status,
                "prediction": body.get("prediction"),
                "probability": body.get("probability_high_quality"),
                "latency": duration,
            }
    except urllib.error.HTTPError as e:
        results[idx] = {"status": e.code, "error": str(e), "latency": time.time() - start}
    except Exception as e:
        results[idx] = {"status": 0, "error": str(e), "latency": time.time() - start}


def run_load_test(url, n_requests, concurrency):
    """Run load test with specified concurrency."""
    print(f"\n{'='*60}")
    print(f"  Load Test — Wine Quality API")
    print(f"  URL: {url}")
    print(f"  Requests: {n_requests}")
    print(f"  Concurrency: {concurrency}")
    print(f"{'='*60}\n")

    results = [None] * n_requests
    start_time = time.time()

    # Process in batches of `concurrency`
    for batch_start in range(0, n_requests, concurrency):
        batch_end = min(batch_start + concurrency, n_requests)
        threads = []
        for i in range(batch_start, batch_end):
            sample = generate_sample()
            t = threading.Thread(target=send_request, args=(url, sample, results, i))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

        done = batch_end
        print(f"  Progress: {done}/{n_requests} ({done/n_requests*100:.0f}%)", end="\r")

    total_time = time.time() - start_time
    print(f"\n\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")

    # Analyze results
    latencies = [r["latency"] for r in results if r]
    statuses = Counter(r.get("status", 0) for r in results if r)
    predictions = Counter(r.get("prediction") for r in results if r and "prediction" in r)
    errors = sum(1 for r in results if r and r.get("status", 0) != 200)

    print(f"  Total time:      {total_time:.2f}s")
    print(f"  Throughput:      {n_requests / total_time:.1f} req/s")
    print(f"  Success:         {n_requests - errors}/{n_requests}")
    print(f"  Errors:          {errors}/{n_requests}")
    print(f"  Avg latency:     {sum(latencies)/len(latencies)*1000:.1f}ms")
    print(f"  Min latency:     {min(latencies)*1000:.1f}ms")
    print(f"  Max latency:     {max(latencies)*1000:.1f}ms")
    print(f"  P50 latency:     {sorted(latencies)[len(latencies)//2]*1000:.1f}ms")
    p95_idx = int(len(latencies) * 0.95)
    print(f"  P95 latency:     {sorted(latencies)[p95_idx]*1000:.1f}ms")
    print(f"  Status codes:    {dict(statuses)}")
    print(f"  Predictions:     Low={predictions.get(0,0)} High={predictions.get(1,0)}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load test Wine Quality API")
    parser.add_argument("--url", default="http://localhost:5000", help="API base URL")
    parser.add_argument("--requests", type=int, default=100, help="Number of requests")
    parser.add_argument("--concurrency", type=int, default=5, help="Concurrent requests")
    args = parser.parse_args()

    run_load_test(args.url, args.requests, args.concurrency)
