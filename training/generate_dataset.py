import numpy as np
import pandas as pd


def generate_synthetic_dataset(
    n_patients: int = 50,
    readings_per_patient: int = 100,
    window_size: int = 12,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    for patient_id in range(n_patients):
        base_hr = rng.uniform(60, 100)
        base_o2 = rng.uniform(94, 99)
        base_activity = rng.uniform(20, 80)

        risk_label = rng.choice([0, 1], p=[0.7, 0.3])

        hr_trend = rng.uniform(-2, 2) if risk_label == 0 else rng.uniform(1, 5)
        o2_trend = (
            rng.uniform(-0.5, 0.5) if risk_label == 0 else rng.uniform(-2, -0.5)
        )
        act_trend = rng.uniform(-1, 1) if risk_label == 0 else rng.uniform(-5, -1)

        for i in range(readings_per_patient):
            hr = base_hr + hr_trend * i / readings_per_patient + rng.normal(0, 2)
            o2 = base_o2 + o2_trend * i / readings_per_patient + rng.normal(0, 0.5)
            act = max(
                0,
                base_activity
                + act_trend * i / readings_per_patient
                + rng.normal(0, 5),
            )
            rows.append(
                {
                    "heartRate": round(hr, 1),
                    "oxygen": round(max(80, min(100, o2)), 1),
                    "activity": round(act, 1),
                    "label": risk_label,
                }
            )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate_synthetic_dataset()
    df.to_csv("dataset_synthetic.csv", index=False)
    print(f"Generated {len(df)} rows")
