"""Generate wide-terrain eval datasets: hospital ops, two-table e-commerce,
wide sensor telemetry, personal banking. Each generator returns (df, truths).
Run:  python scripts/gen_wide_data.py  -> writes data/eval_wide/*.csv
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("data/eval_wide")


def gen_hospital(n=900, seed=7):
    rng = random.Random(seed)
    depts = ["cardiology", "neurology", "oncology", "pediatrics", "emergency"]
    w = [0.25, 0.2, 0.15, 0.2, 0.2]
    rows = []
    for i in range(n):
        d = rng.choices(depts, weights=w)[0]
        los = max(1, int(rng.expovariate(1 / (4 if d != "oncology" else 7))))
        # oncology: longer stays + higher readmit
        readmit = rng.random() < (0.24 if d == "oncology" else 0.11)
        rows.append({
            "patient_id": 5000 + i,
            "department": d,
            "length_of_stay_days": los,
            "readmitted_30d": "yes" if readmit else "no",
            "treatment_cost": round(los * rng.uniform(1800, 3200), 2),
            "age": rng.randint(18, 90),
        })
    df = pd.DataFrame(rows)
    dup_idx = df.sample(n=15, random_state=seed).index
    df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)   # 15 planted dupes
    truths = {
        "top_readmit_dept": ("oncology", round(df[df.department == "oncology"].readmitted_30d.eq("yes").mean() * 100, 1)),
        "median_los": float(df.length_of_stay_days.median()),
    }
    return df, truths


def gen_ecommerce(n=1400, seed=11):
    rng = random.Random(seed)
    customers = pd.DataFrame({
        "customer_id": [f"C{i:04d}" for i in range(200)],
        "segment": rng.choices(["premium", "standard", "basic"], weights=[0.2, 0.5, 0.3], k=200),
        "signup_year": [rng.choice([2021, 2022, 2023]) for _ in range(200)],
    })
    cat_price = {"electronics": 420.0, "home": 130.0, "beauty": 45.0, "sports": 80.0}
    rows = []
    for i in range(n):
        c = customers.sample(1, random_state=rng.randrange(10**6)).iloc[0]
        cat = rng.choice(list(cat_price))
        qty = rng.randint(1, 5)
        price = round(cat_price[cat] * rng.uniform(0.8, 1.2), 2)
        rows.append({
            "order_id": f"O{i:05d}",
            "customer_id": c.customer_id,
            "category": cat,
            "quantity": qty,
            "unit_price": price,
            "returned": "yes" if rng.random() < (0.18 if cat == "home" else 0.07) else "no",
        })
    orders = pd.DataFrame(rows)
    truths = {
        "top_rev_segment": None,   # computed below
        "top_return_cat": ("home", round(orders[orders.category == "home"].returned.eq("yes").mean() * 100, 1)),
    }
    m = orders.merge(customers, on="customer_id")
    m["rev"] = m.unit_price * m.quantity
    rev = m.groupby("segment").rev.sum()
    truths["top_rev_segment"] = (rev.idxmax(), round(float(rev.max()), 2))
    return orders, customers, truths


def gen_sensors(n=1200, seed=13):
    rng = np.random.default_rng(seed)
    machines = ["press-A", "press-B", "lathe-C", "lathe-D", "mixer-E"]
    df = pd.DataFrame({
        "ts": pd.date_range("2024-01-01", periods=n, freq="15min"),
        "machine_id": rng.choice(machines, n),
        "temperature_c": np.round(rng.normal(72, 8, n), 2),
        "vibration_hz": np.round(rng.gamma(2.0, 1.8, n), 3),
        "pressure_bar": np.round(rng.normal(5.2, 0.6, n), 2),
        "rpm": rng.integers(900, 2400, n),
        "power_kw": np.round(rng.normal(18, 3, n), 2),
        "status": rng.choice(["ok", "warn", "fault"], n, p=[0.9, 0.07, 0.03]),
    })
    # plant a heat spike tied to one machine
    mask = df.machine_id == "press-B"
    df.loc[mask, "temperature_c"] = (df.loc[mask, "temperature_c"] + 9).round(2)
    truths = {
        "hottest_machine": ("press-B", round(float(df.groupby("machine_id").temperature_c.mean().max()), 2)),
        "fault_pct": round(float(df.status.eq("fault").mean() * 100), 2),
    }
    return df, truths


def gen_bank(n=1000, seed=17):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        bal = round(rng.lognormvariate(8.2, 1.1), 2)
        overdraft = rng.random() < (0.3 if bal < 500 else 0.04)
        rows.append({
            "account_id": 90000 + i,
            "balance": bal,
            "account_type": rng.choices(["checking", "savings", "credit"], weights=[0.5, 0.35, 0.15])[0],
            "overdraft_flag": 1 if overdraft else 0,
            "monthly_fee": round(rng.uniform(0, 25), 2),
            "branch_city": rng.choice(["springfield", "riverdale", "lakeside", "hillcrest"]),
        })
    df = pd.DataFrame(rows)
    truths = {
        "overdraft_lowbal_pct": round(float(df[df.balance < 500].overdraft_flag.mean() * 100), 1),
        "top_balance_type": None,
    }
    bal = df.groupby("account_type").balance.mean()
    truths["top_balance_type"] = (bal.idxmax(), round(float(bal.max()), 2))
    return df, truths


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    hosp, t_h = gen_hospital()
    orders, customers, t_e = gen_ecommerce()
    sensors, t_s = gen_sensors()
    bank, t_b = gen_bank()
    hosp.to_csv(OUT / "hospital.csv", index=False)
    orders.to_csv(OUT / "orders.csv", index=False)
    customers.to_csv(OUT / "customers.csv", index=False)
    sensors.to_csv(OUT / "sensors.csv", index=False)
    bank.to_csv(OUT / "bank.csv", index=False)
    print(f"wrote {OUT}/*.csv  hospital={len(hosp)} orders={len(orders)}"
          f" customers={len(customers)} sensors={len(sensors)} bank={len(bank)}")
    for k, v in {**t_h, **t_e, **t_s, **t_b}.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
