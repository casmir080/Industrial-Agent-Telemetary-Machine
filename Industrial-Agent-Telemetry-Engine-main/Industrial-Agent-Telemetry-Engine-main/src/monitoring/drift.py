import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from src.telemetry.dataset import CMAPSSDataPipeline

def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Population Stability Index between a reference and current distribution.

    PSI < 0.1  → No significant drift
    PSI < 0.25 → Moderate drift — monitor closely
    PSI >= 0.25 → Significant drift — retraining recommended
    """
    # Build breakpoints from reference distribution only
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))

    # Force unique breakpoints to avoid zero-width bins
    breakpoints = np.unique(breakpoints)

    # Extend edges to capture all actual values without clipping
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    actual_counts, _ = np.histogram(actual, bins=breakpoints)

    # Convert to proportions
    expected_perc = expected_counts / len(expected)
    actual_perc = actual_counts / len(actual)

    # Avoid log(0) by clipping to small epsilon
    expected_perc = np.clip(expected_perc, 1e-6, None)
    actual_perc = np.clip(actual_perc, 1e-6, None)

    psi = np.sum(
        (actual_perc - expected_perc)
        * np.log(actual_perc / expected_perc)
    )

    return round(float(psi), 6)


def interpret_psi(psi: float) -> str:
    if psi < 0.1:
        return "STABLE"
    elif psi < 0.25:
        return "MODERATE_DRIFT"
    return "SIGNIFICANT_DRIFT — RETRAINING RECOMMENDED"


class PSIDriftEngine:
    """
    Continuously tracks feature-level PSI between the training reference
    distribution and an incoming production window.
    """

    def __init__(self, data_dir: str = "data/raw"):
        self.pipeline = CMAPSSDataPipeline(
            data_dir=data_dir,
            sequence_length=30
        )
        self.reference_stats: dict = {}
        self.report_log: list = []

    def fit_reference(self):
        """
        Fits the reference distribution from the training set.
        Extracts per-feature distributions across all training windows.
        """
        train_loader, _ = self.pipeline.prepare_data(batch_size=512)

        all_features = []

        for x_batch, _ in train_loader:
            # (batch, seq, features) → (batch*seq, features)
            flat = x_batch.numpy().reshape(-1, x_batch.shape[-1])
            all_features.append(flat)

        reference = np.vstack(all_features)

        for i in range(reference.shape[1]):
            self.reference_stats[f"sensor_{i+1}"] = reference[:, i]

        print(
            f"[PSI] Reference fitted on "
            f"{reference.shape[0]:,} samples x "
            f"{reference.shape[1]} features."
        )

        return self

    def compute_drift_report(
        self,
        production_window: np.ndarray
    ) -> dict:
        """
        Compares a production window against the reference distribution.

        production_window shape:
            (N, num_features)
        """
        if not self.reference_stats:
            raise RuntimeError(
                "Call fit_reference() before compute_drift_report()."
            )

        results = {}
        drift_detected = False

        for i in range(production_window.shape[1]):
            sensor_key = f"sensor_{i+1}"

            psi_val = compute_psi(
                expected=self.reference_stats[sensor_key],
                actual=production_window[:, i]
            )

            label = interpret_psi(psi_val)

            results[sensor_key] = {
                "psi": psi_val,
                "status": label
            }

            if psi_val >= 0.25:
                drift_detected = True

        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "drift_detected": drift_detected,
            "overall_status": (
                "DRIFT_ALERT"
                if drift_detected
                else "STABLE"
            ),
            "sensors": results
        }

        self.report_log.append(report)

        return report

    def summary(self, report: dict):
        """Prints a clean CLI summary."""
        print(f"\n[DRIFT REPORT] {report['timestamp']}")
        print(f"  Overall Status: {report['overall_status']}")

        drifted = [
            k for k, v in report["sensors"].items()
            if v["psi"] >= 0.1
        ]

        stable = [
            k for k, v in report["sensors"].items()
            if v["psi"] < 0.1
        ]

        print(f"  Stable sensors: {len(stable)}/24")
        print(f"  Drifting sensors: {len(drifted)}/24")

        if drifted:
            print("  Top drifters:")

            top = sorted(
                report["sensors"].items(),
                key=lambda x: x[1]["psi"],
                reverse=True
            )[:5]

            for sensor, data in top:
                print(
                    f"    {sensor}: "
                    f"PSI={data['psi']:.4f} "
                    f"→ {data['status']}"
                )


if __name__ == "__main__":
    print("Initializing PSI Drift Engine Verification...")

    try:
        engine = PSIDriftEngine()
        engine.fit_reference()

        # Test 1: Stable distribution
        print("\n[TEST 1] Stable window:")
        stable_window = np.random.normal(
            0,
            1,
            (500, 24)
        )

        report1 = engine.compute_drift_report(
            stable_window
        )
        engine.summary(report1)

        # Test 2: Drifted distribution
        print("\n[TEST 2] Drifted window:")
        drifted_window = np.random.normal(
            3,
            2,
            (500, 24)
        )

        report2 = engine.compute_drift_report(
            drifted_window
        )
        engine.summary(report2)

        print("\n[SUCCESS] PSI Drift Engine verified.")

    except Exception as e:
        import traceback

        print(f"[ERROR] {e}")
        traceback.print_exc()