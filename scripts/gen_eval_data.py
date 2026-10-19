"""Generate domain-diverse eval datasets with planted, verifiable patterns.

Usage:  .venv/Scripts/python scripts/gen_eval_data.py [outdir]

Creates in outdir (default data/eval):
  churn.csv         telecom-style churn (800 rows)
  movies.csv        movie financials + ratings (600 rows)
  subscriptions.csv monthly SaaS metrics with a churn spike + a MISSING month
  employees.csv     HR payroll with a planted gender pay gap + dupes + nulls

Every dataset ships with GROUND TRUTH functions (gt_* below) the eval harness
recomputes with pandas — the same numbers Jane must find.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/eval")
out_dir.mkdir(parents=True, exist_ok=True)

rng = random.Random(20260929)


# ---------------------------------------------------------------------------
# churn — telecom style
# ---------------------------------------------------------------------------

def gen_churn(path: Path) -> None:
    contracts = ["month-to-month", "one-year", "two-year"]
    payments = ["credit card", "bank transfer", "electronic check", "mailed check"]
    internet = ["fiber", "dsl", "no"]
    rows = ["customer_id,tenure_months,monthly_charges,total_charges,contract,"
            "payment_method,internet_service,support_tickets,senior_citizen,churned"]
    for i in range(800):
        contract = rng.choices(contracts, weights=[55, 25, 20])[0]
        tenure = rng.randint(1, 72)
        charges = round(rng.uniform(20, 115), 2)
        total = round(tenure * charges * rng.uniform(0.97, 1.03), 2)
        if rng.random() < 0.02:
            total = ""                      # 2% missing total_charges
        # churn drivers: month-to-month + high charges + many tickets
        p = 0.05 + (0.35 if contract == "month-to-month" else 0.0) \
            + (0.15 if charges > 90 else 0.0) + min(tickets, 4) * 0.04 \
            if (tickets := rng.randint(0, 6)) is not None else 0.05
        churned = "Yes" if rng.random() < p else "No"
        rows.append(",".join(map(str, [
            f"C{i:04d}", tenure, charges, total, contract,
            rng.choice(payments), rng.choice(internet), tickets,
            rng.choice([0, 0, 0, 1]), churned])))
    path.write_text("\n".join(rows), encoding="utf-8")


def gt_churn(df):
    import pandas as pd
    d = df.copy()
    out = {}
    out["churn_rate"] = round((d["churned"] == "Yes").mean(), 4)
    by_contract = d.groupby("contract")["churned"].apply(lambda s: (s == "Yes").mean())
    out["worst_contract"] = by_contract.idxmax()
    out["worst_contract_rate"] = round(by_contract.max(), 4)
    out["n_month_to_month"] = int((d["contract"] == "month-to-month").sum())
    med = d.groupby("churned")["monthly_charges"].median()
    out["median_charges_churned"] = round(float(med.get("Yes", float("nan"))), 2)
    out["median_charges_stayed"] = round(float(med.get("No", float("nan"))), 2)
    return out


# ---------------------------------------------------------------------------
# movies — financials + ratings, blockbuster outliers
# ---------------------------------------------------------------------------

def gen_movies(path: Path) -> None:
    genres = ["action", "comedy", "drama", "horror", "animation", "sci-fi"]
    rows = ["title,genre,year,budget_m,revenue_m,rating,votes,language"]
    for i in range(600):
        genre = rng.choice(genres)
        year = rng.randint(1990, 2024)
        budget = round(rng.uniform(5, 250), 1)
        mult = {"action": 2.4, "animation": 3.0, "horror": 4.5, "comedy": 2.2,
                "drama": 1.6, "sci-fi": 2.6}[genre]
        revenue = round(budget * mult * rng.uniform(0.3, 1.9), 1)
        if rng.random() < 0.03:
            revenue = round(budget * rng.uniform(8, 14), 1)   # blockbuster outliers
        rating = round(min(9.3, max(4.2, rng.gauss(6.6, 0.9))), 1)
        votes = int(rng.lognormvariate(10.5, 1.3)) + 500
        rows.append(",".join(map(str, [
            f"Movie #{i:03d}", genre, year, budget, revenue, rating, votes,
            rng.choice(["english", "english", "english", "korean", "spanish", "french"])])))
    path.write_text("\n".join(rows), encoding="utf-8")


def gt_movies(df):
    d = df.copy()
    out = {}
    out["n_movies"] = len(d)
    out["max_revenue_movie_idx"] = int(d["revenue_m"].idxmax())       # 'Movie #NNN'
    top = d.sort_values("revenue_m", ascending=False).head(5)
    out["top5_by_revenue"] = top["title"].tolist()
    by_genre = d.groupby("genre")["rating"].mean()
    out["best_genre_by_rating"] = by_genre.idxmax()
    out["best_genre_rating"] = round(float(by_genre.max()), 2)
    out["n_rating_over_8"] = int((d["rating"] > 8.0).sum())
    out["most_releases_year"] = int(d["year"].value_counts().idxmax())
    out["horror_roi"] = round(
        float(d[d["genre"] == "horror"]["revenue_m"].sum()
              / d[d["genre"] == "horror"]["budget_m"].sum()), 2)
    return out


# ---------------------------------------------------------------------------
# subscriptions — monthly SaaS, churn spike in 2024-03, MISSING 2023-06
# ---------------------------------------------------------------------------

def gen_subscriptions(path: Path) -> None:
    months = pd_months = None
    import datetime as dt
    start = dt.date(2022, 1, 1)
    all_months = [start.replace(day=1)]
    while len(all_months) < 36:
        m = all_months[-1]
        nxt = dt.date(m.year + (m.month == 12), 1 if m.month == 12 else m.month + 1, 1)
        all_months.append(nxt)
    active, new, churned, rows = 1200, 0, 0, \
        ["month,new_subs,churned_subs,active_subs,revenue_k"]
    for m in all_months:
        if m.isoformat() == "2023-06-01":
            continue                                   # the planted gap
        new = rng.randint(40, 120)
        churn = rng.randint(15, 45)
        if m.isoformat() == "2024-03-01":
            churn = 210                                # the planted spike
        active = active + new - churn
        rows.append(f"{m.isoformat()[:7]},{new},{churn},{active},{round(active*0.049,1)}")
    path.write_text("\n".join(rows), encoding="utf-8")


def gt_subscriptions(df):
    d = df.copy()
    d["churned_subs"] = d["churned_subs"].astype(int)
    out = {}
    spike = d.loc[d["churned_subs"].idxmax()]
    out["spike_month"] = str(spike["month"])
    out["spike_churned"] = int(spike["churned_subs"])
    out["months_present"] = d["month"].nunique()
    out["missing_month"] = "2023-06"
    out["last_active"] = int(d.iloc[-1]["active_subs"])
    out["avg_new_subs"] = round(float(d["new_subs"].mean()), 1)
    return out


# ---------------------------------------------------------------------------
# employees — HR payroll with pay gap, dupes, nulls
# ---------------------------------------------------------------------------

def gen_employees(path: Path) -> None:
    depts = ["engineering", "sales", "marketing", "support", "finance", "hr"]
    cities = ["berlin", "london", "lisbon", "toronto"]
    rows = ["emp_id,name,department,gender,salary,years_at_company,age,"
            "promotion_last_year,city,manager_id"]
    for i in range(400):
        dept = rng.choice(depts)
        gender = rng.choices(["female", "male"], weights=[45, 55])[0]
        base = {"engineering": 85, "sales": 62, "marketing": 58,
                "support": 45, "finance": 66, "hr": 52}[dept]
        years = rng.randint(0, 15)
        salary = round((base + years * 2.2 + rng.uniform(-6, 6))
                       * (0.92 if gender == "female" else 1.0) * 1000, 0)
        if rng.random() < 0.02:
            salary = ""                                # 2% null salary
        rows.append(",".join(map(str, [
            i, f"Employee {i:03d}", dept, gender, int(salary) if salary else "",
            years, rng.randint(22, 60), rng.choice([0, 0, 0, 1]),
            rng.choice(cities), rng.choice([i, rng.randint(0, 399)])])))
    dup = [rows[rng.randint(1, 400)] for _ in range(12)]
    path.write_text("\n".join(rows + dup), encoding="utf-8")


def gt_employees(df):
    d = df.copy()
    d = d.drop_duplicates()
    out = {}
    out["rows_raw"] = 400
    out["dupes"] = 12
    by_dept = d.groupby("department")["salary"].median()
    out["top_dept_by_median_salary"] = by_dept.idxmax()
    out["top_dept_median"] = round(float(by_dept.max()), 0)
    gap = d.groupby("gender")["salary"].mean()
    out["male_mean_salary"] = round(float(gap["male"]), 0)
    out["female_mean_salary"] = round(float(gap["female"]), 0)
    out["pay_gap_pct"] = round(float(1 - gap["female"] / gap["male"]) * 100, 1)
    out["headcount_engineering"] = int((d["department"] == "engineering").sum())
    return out


GENERATORS = {
    "churn.csv": (gen_churn, gt_churn),
    "movies.csv": (gen_movies, gt_movies),
    "subscriptions.csv": (gen_subscriptions, gt_subscriptions),
    "employees.csv": (gen_employees, gt_employees),
}


def main() -> int:
    import pandas as pd
    for name, (gen, gt) in GENERATORS.items():
        p = out_dir / name
        gen(p)
        df = pd.read_csv(p)
        truth = gt(df)
        print(f"{name:20s} {len(df):>5} rows × {len(df.columns)} cols")
        for k, v in truth.items():
            print(f"    {k:28s} = {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
