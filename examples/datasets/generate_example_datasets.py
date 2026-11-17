"""Reproducible generator for UAP multi-discipline medium, large, wide, and massive CSV benchmark datasets.

Every dataset embeds realistic domain signal (linear, nonlinear, interaction, seasonal, and latent cluster
structure) together with realistic anomalies, extreme outliers, corrupt sensor/entry records, missing values,
messy categorical strings, and exact duplicates so users can thoroughly test cleaning, Studio 02 anomaly and
cluster screening, Studio 03 statistical diagnostics & baselines, Studio 05 feature engineering, Studio 06
AutoML/Deep Learning equations & future target prediction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent
FUTURE_DIR = OUT_DIR / "future_incoming"


def _inject_missing(df: pd.DataFrame, cols: list[str], rate: float, rng: np.random.Generator) -> None:
    """Inject realistic missing values into selected predictor columns."""
    n = len(df)
    for col in cols:
        mask = rng.random(n) < rate
        df.loc[mask, col] = np.nan


def _append_duplicates(df: pd.DataFrame, n_dups: int, rng: np.random.Generator) -> pd.DataFrame:
    """Append exact duplicate rows and shuffle deterministically."""
    if n_dups <= 0:
        return df
    dup_idx = rng.choice(len(df), size=min(n_dups, len(df)), replace=False)
    combined = pd.concat([df, df.iloc[dup_idx]], ignore_index=True)
    return combined.sample(frac=1.0, random_state=int(rng.integers(0, 10_000))).reset_index(drop=True)


def generate_healthcare_medium(seed: int = 101) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Medium clinical dataset (4,500 rows x 18 cols) with physiological anomalies & 4 patient clusters."""
    rng = np.random.default_rng(seed)
    n = 4455  # +45 duplicates -> 4,500 rows

    # 4 distinct patient risk cohorts
    cohort = rng.choice([0, 1, 2, 3], size=n, p=[0.30, 0.28, 0.24, 0.18])
    age = np.where(cohort == 0, rng.normal(31, 8, n),
          np.where(cohort == 1, rng.normal(54, 9, n),
          np.where(cohort == 2, rng.normal(66, 8, n), rng.normal(79, 6, n)))).clip(18, 96)
    bmi = np.where(cohort == 1, rng.normal(32.5, 4.5, n), rng.normal(25.8, 4.0, n)).clip(16, 52)
    systolic_bp = np.where(cohort >= 2, rng.normal(148, 18, n), rng.normal(121, 13, n)).clip(85, 210)
    diastolic_bp = (systolic_bp * rng.uniform(0.58, 0.68, n) + rng.normal(0, 4, n)).clip(50, 130)
    heart_rate_bpm = np.where(cohort == 2, rng.normal(92, 14, n), rng.normal(74, 11, n)).clip(45, 155)
    glucose_mg_dl = np.where(cohort == 1, rng.normal(185, 45, n), rng.normal(102, 22, n)).clip(62, 420)
    hba1c_pct = np.clip(4.6 + 0.018 * glucose_mg_dl + rng.normal(0, 0.45, n), 4.2, 13.5)
    creatinine_mg_dl = np.where(cohort == 3, rng.normal(1.85, 0.55, n), rng.normal(0.95, 0.25, n)).clip(0.4, 5.5)
    oxygen_sat_pct = np.where(cohort == 3, rng.normal(93.8, 2.5, n), rng.normal(97.8, 1.3, n)).clip(82, 100)
    prior_admissions = rng.poisson(np.where(cohort == 3, 2.6, np.where(cohort == 2, 1.4, 0.4))).astype(float)
    num_medications = np.round(np.clip(2.0 + 1.8 * cohort + rng.normal(0, 2.0, n), 0, 24))
    lab_procedures = np.round(np.clip(12 + 8 * cohort + rng.normal(0, 6, n), 1, 80))

    departments = np.where(
        cohort == 0,
        rng.choice(["General Medicine", "Orthopedics"], size=n),
        np.where(cohort == 1, rng.choice(["Endocrinology", "Internal Medicine"], size=n),
        np.where(cohort == 2, rng.choice(["Cardiology", "Emergency"], size=n), rng.choice(["Geriatrics", "ICU", "Cardiology"], size=n))),
    )
    admission_type = rng.choice(["Elective", "Urgent", "Emergency"], size=n, p=[0.35, 0.30, 0.35])
    insurance_tier = rng.choice(["Public", "Private", "Employer", "Uninsured"], size=n, p=[0.38, 0.32, 0.22, 0.08])

    # True nonlinear signal for both Length-of-Stay (regression) and 30-day Readmission (binary classification)
    los = (
        1.2
        + 0.035 * age
        + 0.012 * np.maximum(0, glucose_mg_dl - 110)
        + 1.35 * creatinine_mg_dl
        + 0.85 * prior_admissions
        + 0.14 * num_medications
        - 0.22 * (oxygen_sat_pct - 95.0)
        + 1.4 * (admission_type == "Emergency").astype(float)
        + rng.normal(0, 1.1, n)
    ).clip(1.0, 35.0)

    logit = (
        -4.2
        + 0.028 * age
        + 0.55 * prior_admissions
        + 0.72 * creatinine_mg_dl
        + 0.24 * hba1c_pct
        + 0.012 * (systolic_bp - 120)
        - 0.18 * (oxygen_sat_pct - 95)
        + 0.45 * (admission_type == "Emergency").astype(float)
        + 0.002 * (age - 55) * (creatinine_mg_dl - 1.0)
    )
    prob = 1.0 / (1.0 + np.exp(-np.clip(logit, -15, 15)))
    readmitted = (rng.random(n) < prob).astype(int)

    df = pd.DataFrame(
        {
            "patient_id": [f"PAT-{100000 + i}" for i in range(n)],
            "age_years": np.round(age, 1),
            "bmi": np.round(bmi, 2),
            "systolic_bp_mmhg": np.round(systolic_bp, 1),
            "diastolic_bp_mmhg": np.round(diastolic_bp, 1),
            "heart_rate_bpm": np.round(heart_rate_bpm, 1),
            "glucose_mg_dl": np.round(glucose_mg_dl, 1),
            "hba1c_pct": np.round(hba1c_pct, 2),
            "creatinine_mg_dl": np.round(creatinine_mg_dl, 2),
            "oxygen_sat_pct": np.round(oxygen_sat_pct, 1),
            "prior_admissions_12m": prior_admissions,
            "num_medications": num_medications,
            "lab_procedures_count": lab_procedures,
            "department": departments,
            "admission_type": admission_type,
            "insurance_tier": insurance_tier,
            "length_of_stay_days": np.round(los, 2),
            "readmitted_30d": readmitted,
        }
    )

    # Save 50 clean future incoming patient records (no target columns, slight shift + 4 OOD cases)
    future_df = df.iloc[:50].drop(columns=["length_of_stay_days", "readmitted_30d"]).copy().reset_index(drop=True)
    future_df["patient_id"] = [f"FUT-PAT-{i + 1:03d}" for i in range(len(future_df))]
    future_df["age_years"] = np.round(future_df["age_years"] + rng.normal(2.5, 1.5, len(future_df)), 1)
    future_df.loc[46:49, "glucose_mg_dl"] = [460.0, 510.0, 395.0, 480.0]
    future_df.loc[48:49, "creatinine_mg_dl"] = [6.4, 7.1]

    # Inject severe physiological outliers & corrupt data-entry errors (~6% of rows)
    out_idx = rng.choice(n, size=int(n * 0.055), replace=False)
    chunks = np.array_split(out_idx, 5)
    df.loc[chunks[0], "age_years"] = rng.choice([-12.0, -3.0, 185.0, 215.0], size=len(chunks[0]))
    df.loc[chunks[1], "systolic_bp_mmhg"] = rng.uniform(310.0, 460.0, size=len(chunks[1])).round(1)
    df.loc[chunks[1], "heart_rate_bpm"] = rng.choice([0.0, 280.0, 325.0], size=len(chunks[1]))
    df.loc[chunks[2], "glucose_mg_dl"] = rng.uniform(750.0, 1650.0, size=len(chunks[2])).round(1)
    df.loc[chunks[3], "bmi"] = rng.choice([-5.2, 88.4, 112.0], size=len(chunks[3]))
    df.loc[chunks[4], "oxygen_sat_pct"] = rng.choice([12.0, 145.0, 999.0], size=len(chunks[4]))

    # Messy categorical whitespace / casing inconsistencies
    messy_idx = rng.choice(n, size=180, replace=False)
    df.loc[messy_idx[:60], "department"] = "  cardiology "
    df.loc[messy_idx[60:120], "department"] = "EMERGENCY"
    df.loc[messy_idx[120:], "admission_type"] = " urgent "

    _inject_missing(df, ["bmi", "hba1c_pct", "creatinine_mg_dl", "glucose_mg_dl", "insurance_tier"], 0.042, rng)
    df = _append_duplicates(df, 45, rng)
    return df, future_df


def generate_manufacturing_large(seed: int = 202) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Large industrial IoT predictive maintenance dataset (25,000 rows x 22 cols) with sensor spikes & corrupt cluster."""
    rng = np.random.default_rng(seed)
    n = 24880  # +120 duplicates -> 25,000 rows

    regime = rng.choice([0, 1, 2, 3, 4], size=n, p=[0.18, 0.36, 0.22, 0.14, 0.10])
    rpm = np.where(regime == 0, rng.normal(600, 80, n),
          np.where(regime == 1, rng.normal(1500, 120, n),
          np.where(regime == 2, rng.normal(2450, 180, n),
          np.where(regime == 3, rng.normal(1850, 160, n), rng.normal(2850, 210, n))))).clip(100, 4200)
    torque_nm = (18.0 + 0.025 * rpm + np.where(regime >= 3, 18.0, 0.0) + rng.normal(0, 6.0, n)).clip(5, 180)
    bearing_temp_c = (32.0 + 0.018 * rpm + 0.25 * torque_nm + np.where(regime == 2, 16.0, 0.0) + rng.normal(0, 4.5, n)).clip(20, 165)
    ambient_temp_c = rng.normal(24.0, 5.5, n).clip(5, 48)
    vibration_rms = np.where(regime == 4, rng.normal(11.5, 2.4, n),
                    np.where(regime == 3, rng.normal(7.2, 1.5, n), rng.normal(2.4, 0.8, n))).clip(0.1, 30.0)
    acoustic_db = (58.0 + 2.3 * vibration_rms + 0.005 * rpm + rng.normal(0, 2.8, n)).clip(45, 130)
    oil_pressure_bar = np.where(regime == 3, rng.normal(2.6, 0.5, n), rng.normal(4.8, 0.6, n)).clip(0.5, 9.5)
    oil_viscosity_cst = (46.0 - 0.18 * (bearing_temp_c - 40.0) + rng.normal(0, 2.5, n)).clip(12, 85)
    motor_current_a = (4.5 + 0.14 * torque_nm + 0.002 * rpm + rng.normal(0, 1.2, n)).clip(1.0, 55.0)
    voltage_v = rng.normal(400.0, 8.5, n).clip(355, 445)
    power_factor = rng.uniform(0.76, 0.98, n)
    tool_wear_min = np.where(regime >= 3, rng.uniform(140, 260, n), rng.uniform(0, 170, n))
    operating_hours = rng.uniform(50, 18000, n)
    coolant_flow_lpm = np.where(regime == 2, rng.normal(9.5, 1.8, n), rng.normal(15.2, 2.1, n)).clip(2.0, 30.0)
    humidity_pct = rng.uniform(20, 88, n)

    machine_class = rng.choice(["Turbine_A", "CNC_Mill_B", "Hydraulic_Press_C", "Compressor_D"], size=n)
    plant_site = rng.choice(["Munich_DE", "Detroit_US", "Osaka_JP", "Monterrey_MX"], size=n)
    shift = rng.choice(["Day", "Swing", "Night"], size=n, p=[0.45, 0.30, 0.25])

    # Physics-informed wear & failure model
    stress_index = (
        0.055 * (bearing_temp_c - 65.0)
        + 0.42 * vibration_rms
        + 0.018 * tool_wear_min
        - 0.65 * (oil_pressure_bar - 4.0)
        + 0.00018 * operating_hours
        + 0.015 * np.maximum(0.0, motor_current_a - 24.0) ** 1.4
    )
    fail_prob = 1.0 / (1.0 + np.exp(-np.clip(stress_index - 5.2, -14, 14)))
    equipment_failure = (rng.random(n) < fail_prob).astype(int)

    rul_hrs = np.clip(
        480.0 - 42.0 * stress_index - 0.012 * operating_hours + rng.normal(0, 28.0, n),
        1.0,
        650.0,
    )
    rul_hrs = np.where(equipment_failure == 1, rng.uniform(0.5, 18.0, n), rul_hrs)

    failure_mode = np.where(
        equipment_failure == 0,
        "Healthy",
        np.where(
            bearing_temp_c > 98,
            "Thermal_Overload",
            np.where(vibration_rms > 7.5, "Bearing_Spalling", np.where(oil_pressure_bar < 3.0, "Lubrication_Loss", "Tool_Wear_Fracture")),
        ),
    )

    df = pd.DataFrame(
        {
            "machine_id": [f"MCH-{i % 250:03d}" for i in range(n)],
            "machine_class": machine_class,
            "plant_site": plant_site,
            "shift": shift,
            "rpm": np.round(rpm, 1),
            "torque_nm": np.round(torque_nm, 2),
            "bearing_temp_c": np.round(bearing_temp_c, 2),
            "ambient_temp_c": np.round(ambient_temp_c, 2),
            "vibration_rms_mms": np.round(vibration_rms, 3),
            "acoustic_db": np.round(acoustic_db, 2),
            "oil_pressure_bar": np.round(oil_pressure_bar, 2),
            "oil_viscosity_cst": np.round(oil_viscosity_cst, 2),
            "motor_current_a": np.round(motor_current_a, 2),
            "voltage_v": np.round(voltage_v, 2),
            "power_factor": np.round(power_factor, 3),
            "tool_wear_min": np.round(tool_wear_min, 1),
            "operating_hours": np.round(operating_hours, 1),
            "coolant_flow_lpm": np.round(coolant_flow_lpm, 2),
            "humidity_pct": np.round(humidity_pct, 1),
            "remaining_useful_life_hrs": np.round(rul_hrs, 2),
            "equipment_failure": equipment_failure,
            "failure_mode": failure_mode,
        }
    )

    # 100 future incoming telemetry packets (no targets, including rising wear & 6 sensor anomalies)
    future_df = (
        df.iloc[:100]
        .drop(columns=["remaining_useful_life_hrs", "equipment_failure", "failure_mode"])
        .copy()
        .reset_index(drop=True)
    )
    future_df["tool_wear_min"] = np.round(future_df["tool_wear_min"] + 25.0, 1)
    future_df.loc[94:99, "vibration_rms_mms"] = [19.4, 24.8, 28.5, 34.2, 22.1, 29.9]
    future_df.loc[94:99, "bearing_temp_c"] = [132.0, 145.5, 152.0, 168.0, 138.0, 159.0]

    # Inject extreme sensor spikes + a 220-row corrupt stuck-sensor anomaly cluster
    spike_idx = rng.choice(n, size=int(n * 0.045), replace=False)
    s_parts = np.array_split(spike_idx, 4)
    df.loc[s_parts[0], "vibration_rms_mms"] = rng.uniform(85.0, 999.9, size=len(s_parts[0])).round(2)
    df.loc[s_parts[1], "bearing_temp_c"] = rng.choice([-99.0, -40.0, 390.0, 485.0], size=len(s_parts[1]))
    df.loc[s_parts[2], "rpm"] = rng.choice([-1500.0, 0.0, 28500.0], size=len(s_parts[2]))
    df.loc[s_parts[3], "oil_pressure_bar"] = rng.choice([-4.5, 65.0, 95.0], size=len(s_parts[3]))

    corrupt_cluster_idx = rng.choice(n, size=220, replace=False)
    df.loc[corrupt_cluster_idx, ["rpm", "torque_nm", "motor_current_a", "vibration_rms_mms"]] = [9999.0, -99.0, 250.0, 120.0]

    _inject_missing(df, ["oil_viscosity_cst", "acoustic_db", "coolant_flow_lpm", "humidity_pct", "voltage_v"], 0.032, rng)
    df = _append_duplicates(df, 120, rng)
    return df, future_df


def generate_finance_large(seed: int = 303) -> pd.DataFrame:
    """Large credit risk & fraud dataset (35,000 rows x 20 cols) with fat-tail Pareto shocks & fraud ring."""
    rng = np.random.default_rng(seed)
    n = 34850  # +150 duplicates -> 35,000 rows

    seg = rng.choice([0, 1, 2, 3], size=n, p=[0.46, 0.28, 0.20, 0.06])
    annual_income = np.where(
        seg == 0,
        rng.lognormal(11.1, 0.45, n),
        np.where(seg == 1, rng.lognormal(11.6, 0.60, n), np.where(seg == 2, rng.lognormal(10.6, 0.50, n), rng.lognormal(10.2, 0.70, n))),
    ).clip(12_000, 1_500_000)
    loan_amount = (annual_income * rng.uniform(0.08, 0.85, n)).clip(1_000, 950_000)
    credit_score = np.where(seg == 0, rng.normal(745, 42, n),
                   np.where(seg == 1, rng.normal(695, 55, n),
                   np.where(seg == 2, rng.normal(630, 60, n), rng.normal(545, 65, n)))).clip(300, 850)
    debt_to_income = np.where(seg >= 2, rng.beta(3.2, 4.0, n), rng.beta(1.8, 5.5, n)) * 1.15
    credit_utilization = np.where(seg >= 2, rng.beta(4.0, 2.8, n), rng.beta(2.0, 5.0, n))
    credit_history_yrs = np.clip(rng.gamma(3.5, 3.0, n), 0.2, 42.0)
    open_accounts = rng.poisson(np.where(seg == 1, 9.0, 6.0)).astype(float)
    delinquencies_24m = rng.poisson(np.where(seg == 3, 2.2, np.where(seg == 2, 0.7, 0.12))).astype(float)
    tx_velocity_24h = rng.poisson(np.where(seg == 3, 14.0, 3.2)).astype(float)
    ip_geo_distance_km = np.where(seg == 3, rng.exponential(1800, n), rng.exponential(45, n)).clip(0, 15000)
    collateral_ratio = np.clip(rng.normal(0.72, 0.28, n), 0.0, 2.0)
    interest_rate_pct = np.clip(22.0 - 0.022 * credit_score + 3.2 * debt_to_income + rng.normal(0, 1.1, n), 2.5, 32.0)
    savings_balance = (annual_income * rng.beta(1.5, 5.0, n) * 1.4).clip(0, 2_000_000)

    product_type = rng.choice(["Mortgage", "Auto_Loan", "Personal_Credit", "SME_Working_Capital", "Card_Line"], size=n)
    channel = rng.choice(["Web", "Mobile_App", "Branch", "Partner_API"], size=n, p=[0.35, 0.35, 0.18, 0.12])
    region = rng.choice(["North_America", "Western_Europe", "APAC", "LATAM"], size=n)
    employment_status = rng.choice(["Salaried", "Self_Employed", "Corporate_Exec", "Contract", "Unemployed"], size=n, p=[0.55, 0.18, 0.10, 0.12, 0.05])

    risk_logit = (
        -3.6
        - 0.011 * (credit_score - 650)
        + 2.4 * debt_to_income
        + 1.6 * credit_utilization
        + 0.75 * delinquencies_24m
        + 0.12 * tx_velocity_24h
        + 0.0005 * ip_geo_distance_km
        - 0.85 * collateral_ratio
        + 0.95 * (employment_status == "Unemployed").astype(float)
    )
    pd_prob = 1.0 / (1.0 + np.exp(-np.clip(risk_logit, -15, 15)))
    default_or_fraud = (rng.random(n) < pd_prob).astype(int)
    lgd = np.clip(0.55 - 0.25 * collateral_ratio + rng.normal(0, 0.08, n), 0.05, 0.98)
    expected_loss = np.where(default_or_fraud == 1, loan_amount * lgd, loan_amount * pd_prob * lgd * 0.35)

    df = pd.DataFrame(
        {
            "account_id": [f"FIN-{1000000 + i}" for i in range(n)],
            "product_type": product_type,
            "channel": channel,
            "region": region,
            "employment_status": employment_status,
            "annual_income_usd": np.round(annual_income, 2),
            "loan_amount_usd": np.round(loan_amount, 2),
            "savings_balance_usd": np.round(savings_balance, 2),
            "credit_score": np.round(credit_score, 0),
            "debt_to_income": np.round(debt_to_income, 4),
            "credit_utilization": np.round(credit_utilization, 4),
            "credit_history_years": np.round(credit_history_yrs, 2),
            "open_accounts": open_accounts,
            "delinquencies_24m": delinquencies_24m,
            "tx_velocity_24h": tx_velocity_24h,
            "ip_geo_distance_km": np.round(ip_geo_distance_km, 1),
            "collateral_ratio": np.round(collateral_ratio, 3),
            "interest_rate_pct": np.round(interest_rate_pct, 2),
            "expected_loss_usd": np.round(expected_loss, 2),
            "default_or_fraud_flag": default_or_fraud,
        }
    )

    # Inject heavy fat-tail Pareto shocks, impossible credit scores, negative loans & synthetic fraud ring
    shock_idx = rng.choice(n, size=int(n * 0.045), replace=False)
    sp = np.array_split(shock_idx, 4)
    df.loc[sp[0], "annual_income_usd"] = np.round(rng.pareto(1.4, size=len(sp[0])) * 2_500_000 + 5_000_000, 2)
    df.loc[sp[1], "credit_score"] = rng.choice([50.0, 120.0, 995.0, 1250.0], size=len(sp[1]))
    df.loc[sp[2], "loan_amount_usd"] = rng.uniform(-85_000.0, -1_500.0, size=len(sp[2])).round(2)
    df.loc[sp[3], "debt_to_income"] = rng.uniform(5.5, 24.0, size=len(sp[3])).round(3)
    df.loc[sp[3], "tx_velocity_24h"] = rng.integers(120, 480, size=len(sp[3])).astype(float)

    _inject_missing(df, ["savings_balance_usd", "collateral_ratio", "credit_history_years", "employment_status"], 0.038, rng)
    return _append_duplicates(df, 150, rng)


def generate_energy_timeseries_large(seed: int = 404) -> pd.DataFrame:
    """Large hourly smart-grid energy demand & climate time series (40,000 rows x 16 cols) with blackouts & price spikes."""
    rng = np.random.default_rng(seed)
    n = 39900  # +100 duplicates -> 40,000 rows

    ts = pd.date_range("2022-01-01 00:00:00", periods=n, freq="h")
    hour = ts.hour.to_numpy(dtype=float)
    dow = ts.dayofweek.to_numpy(dtype=float)
    doy = ts.dayofyear.to_numpy(dtype=float)
    t_idx = np.arange(n, dtype=float)

    # Seasonal weather physics
    ambient_temp_c = (
        14.0
        + 13.5 * np.sin(2 * np.pi * (doy - 105) / 365.25)
        + 4.2 * np.sin(2 * np.pi * (hour - 9) / 24.0)
        + rng.normal(0, 2.8, n)
    )
    solar_irradiance = np.maximum(
        0.0,
        (780.0 + 260.0 * np.sin(2 * np.pi * (doy - 80) / 365.25)) * np.sin(np.pi * np.clip((hour - 5.5) / 13.5, 0, 1))
        + rng.normal(0, 55, n),
    )
    wind_speed_ms = np.clip(rng.weibull(2.1, n) * 7.8, 0.2, 28.0)
    humidity_pct = np.clip(62.0 - 0.8 * (ambient_temp_c - 14.0) + rng.normal(0, 10.0, n), 12, 99)
    cloud_cover_pct = np.clip(100.0 - 0.08 * solar_irradiance + rng.normal(0, 14, n), 0, 100)
    is_weekend = (dow >= 5).astype(int)
    industrial_index = 100.0 + 0.00055 * t_idx + 2.5 * np.sin(2 * np.pi * doy / 365.25) + rng.normal(0, 1.2, n)
    ev_charging_mw = np.clip(
        (25.0 + 0.0018 * t_idx) * (1.0 + 0.75 * np.exp(-0.5 * ((hour - 20.0) / 2.8) ** 2)) + rng.normal(0, 4.0, n),
        2.0,
        240.0,
    )
    solar_wind_gen_mw = np.clip(0.32 * solar_irradiance + 14.5 * (wind_speed_ms**1.25) + rng.normal(0, 15, n), 5.0, 950.0)

    # Nonlinear U-shaped temperature load + daily/weekly seasonality + secular growth
    temp_load = 1.65 * (ambient_temp_c - 18.0) ** 2
    daily_cycle = 165.0 * np.sin(2 * np.pi * (hour - 7) / 24.0) + 55.0 * np.cos(4 * np.pi * (hour - 18) / 24.0)
    grid_load_mw = (
        920.0
        + 0.0045 * t_idx
        + temp_load
        + daily_cycle
        - 125.0 * is_weekend
        + 1.05 * ev_charging_mw
        + 2.4 * (industrial_index - 100.0)
        + rng.normal(0, 26.0, n)
    ).clip(350.0, 2600.0)

    net_demand_mw = np.maximum(50.0, grid_load_mw - solar_wind_gen_mw)
    spot_price = np.clip(
        28.0 + 0.045 * net_demand_mw + 0.00009 * np.maximum(0.0, net_demand_mw - 900.0) ** 2 + rng.normal(0, 8.0, n),
        -15.0,
        650.0,
    )
    price_spike_alert = (spot_price > 135.0).astype(int)
    grid_zone = rng.choice(["North_Interconnect", "Central_Hub", "Coastal_South", "Alpine_East"], size=n)

    df = pd.DataFrame(
        {
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "grid_zone": grid_zone,
            "hour_of_day": hour.astype(int),
            "day_of_week": dow.astype(int),
            "is_weekend": is_weekend,
            "ambient_temp_c": np.round(ambient_temp_c, 2),
            "humidity_pct": np.round(humidity_pct, 1),
            "wind_speed_ms": np.round(wind_speed_ms, 2),
            "solar_irradiance_wm2": np.round(solar_irradiance, 1),
            "cloud_cover_pct": np.round(cloud_cover_pct, 1),
            "industrial_index": np.round(industrial_index, 2),
            "ev_charging_mw": np.round(ev_charging_mw, 2),
            "solar_wind_generation_mw": np.round(solar_wind_gen_mw, 2),
            "spot_price_usd_mwh": np.round(spot_price, 2),
            "grid_load_mw": np.round(grid_load_mw, 2),
            "price_spike_alert": price_spike_alert,
        }
    )

    # Inject grid blackouts, negative metering errors, extreme weather & wholesale price shocks (~4.2%)
    anom_idx = rng.choice(n, size=int(n * 0.042), replace=False)
    ap = np.array_split(anom_idx, 4)
    df.loc[ap[0], "grid_load_mw"] = rng.choice([0.0, -320.0, -85.0], size=len(ap[0]))
    df.loc[ap[1], "spot_price_usd_mwh"] = rng.uniform(1800.0, 5200.0, size=len(ap[1])).round(2)
    df.loc[ap[2], "ambient_temp_c"] = rng.choice([-68.0, 74.0, 112.0], size=len(ap[2]))
    df.loc[ap[3], "wind_speed_ms"] = rng.uniform(95.0, 240.0, size=len(ap[3])).round(1)

    _inject_missing(df, ["solar_irradiance_wm2", "wind_speed_ms", "humidity_pct", "ev_charging_mw"], 0.030, rng)
    # Append 100 exact duplicate telemetry rows (keep chronological ordering mostly intact)
    dup_rows = df.iloc[rng.choice(n, size=100, replace=False)].copy()
    return pd.concat([df, dup_rows], ignore_index=True)


def generate_retail_massive(seed: int = 505) -> pd.DataFrame:
    """Massive e-commerce & retail customer analytics dataset (100,000 rows x 22 cols) with bots & whale outliers."""
    rng = np.random.default_rng(seed)
    n = 99700  # +300 duplicates -> 100,000 rows

    cohort = rng.choice([0, 1, 2, 3, 4], size=n, p=[0.28, 0.30, 0.22, 0.15, 0.05])
    tenure_months = np.where(cohort == 3, rng.normal(42, 14, n),
                    np.where(cohort == 1, rng.normal(26, 10, n), rng.normal(11, 8, n))).clip(0.5, 96)
    orders_12m = np.where(cohort == 3, rng.poisson(24, n),
                 np.where(cohort == 1, rng.poisson(12, n),
                 np.where(cohort == 2, rng.poisson(5, n), rng.poisson(2.5, n)))).astype(float)
    avg_order_value = np.where(cohort == 3, rng.lognormal(5.5, 0.45, n),
                      np.where(cohort == 1, rng.lognormal(4.5, 0.40, n), rng.lognormal(3.9, 0.50, n))).clip(8, 6500)
    days_since_last = np.where(cohort == 0, rng.exponential(85, n),
                      np.where(cohort == 3, rng.exponential(14, n), rng.exponential(36, n))).clip(0, 420)
    sessions_30d = np.where(cohort == 4, rng.poisson(45, n), rng.poisson(1 + 0.5 * orders_12m)).astype(float)
    avg_session_sec = np.where(cohort == 4, rng.uniform(0.3, 4.0, n), rng.lognormal(5.3, 0.65, n)).clip(0.2, 3600)
    pages_per_session = np.where(cohort == 4, rng.integers(80, 350, n), rng.poisson(5.8, n) + 1).astype(float)
    cart_abandon_rate = np.where(cohort == 0, rng.beta(4.5, 2.2, n), rng.beta(2.0, 4.5, n))
    discount_usage_rate = rng.beta(2.2, 3.8, n)
    return_rate = np.where(cohort == 0, rng.beta(2.5, 6.0, n), rng.beta(1.2, 10.0, n))
    support_tickets_12m = rng.poisson(np.where(cohort == 0, 2.4, 0.6)).astype(float)
    nps_score = np.where(cohort == 3, rng.normal(8.8, 1.1, n),
                np.where(cohort == 0, rng.normal(5.4, 2.0, n), rng.normal(7.5, 1.5, n))).clip(0, 10)
    email_open_rate = np.clip(0.12 + 0.045 * nps_score + rng.normal(0, 0.12, n), 0.0, 1.0)
    loyalty_points = np.round(np.maximum(0.0, orders_12m * avg_order_value * 0.4 + rng.normal(0, 120, n)))
    shipping_delay_days = np.clip(rng.exponential(1.4, n), 0, 21)

    channel = rng.choice(["Mobile_App", "Web", "In_Store", "Marketplace"], size=n, p=[0.42, 0.34, 0.14, 0.10])
    membership_tier = np.where(cohort == 3, "Platinum", np.where(cohort == 1, "Gold", rng.choice(["Silver", "Bronze"], size=n)))
    category_pref = rng.choice(["Electronics", "Apparel", "Home_Living", "Health_Beauty", "Groceries", "Industrial"], size=n)
    country = rng.choice(["US", "DE", "UK", "FR", "JP", "CA", "AU"], size=n)

    churn_logit = (
        -2.4
        + 0.022 * days_since_last
        + 0.55 * support_tickets_12m
        + 1.9 * cart_abandon_rate
        + 2.1 * return_rate
        - 0.36 * nps_score
        - 0.08 * orders_12m
        + 0.14 * shipping_delay_days
        - 0.025 * tenure_months
    )
    churn_prob = 1.0 / (1.0 + np.exp(-np.clip(churn_logit, -15, 15)))
    churned_90d = (rng.random(n) < churn_prob).astype(int)

    customer_ltv = np.clip(
        orders_12m * avg_order_value * (1.0 - 0.55 * return_rate) * (1.6 - 0.75 * churn_prob)
        + 0.08 * loyalty_points
        + rng.normal(0, 120, n),
        5.0,
        150_000.0,
    )

    df = pd.DataFrame(
        {
            "customer_id": [f"CUS-{1000000 + i}" for i in range(n)],
            "country": country,
            "primary_channel": channel,
            "membership_tier": membership_tier,
            "favorite_category": category_pref,
            "tenure_months": np.round(tenure_months, 1),
            "orders_12m": orders_12m,
            "avg_order_value_usd": np.round(avg_order_value, 2),
            "days_since_last_order": np.round(days_since_last, 1),
            "sessions_30d": sessions_30d,
            "avg_session_sec": np.round(avg_session_sec, 1),
            "pages_per_session": pages_per_session,
            "cart_abandon_rate": np.round(cart_abandon_rate, 3),
            "discount_usage_rate": np.round(discount_usage_rate, 3),
            "return_rate": np.round(return_rate, 3),
            "support_tickets_12m": support_tickets_12m,
            "nps_score": np.round(nps_score, 1),
            "email_open_rate": np.round(email_open_rate, 3),
            "loyalty_points": loyalty_points,
            "avg_shipping_delay_days": np.round(shipping_delay_days, 2),
            "customer_ltv_usd": np.round(customer_ltv, 2),
            "churned_90d": churned_90d,
        }
    )

    # Inject bot accounts, whale outliers, impossible negative days & return rates > 1.0 (~4.8%)
    anom_idx = rng.choice(n, size=int(n * 0.048), replace=False)
    ap = np.array_split(anom_idx, 5)
    df.loc[ap[0], "pages_per_session"] = rng.integers(600, 2500, size=len(ap[0])).astype(float)
    df.loc[ap[0], "avg_session_sec"] = rng.uniform(0.05, 0.35, size=len(ap[0])).round(2)
    df.loc[ap[1], "avg_order_value_usd"] = rng.uniform(45_000.0, 185_000.0, size=len(ap[1])).round(2)
    df.loc[ap[2], "days_since_last_order"] = rng.choice([-45.0, -7.0, 9999.0], size=len(ap[2]))
    df.loc[ap[3], "return_rate"] = rng.uniform(1.25, 3.80, size=len(ap[3])).round(2)
    df.loc[ap[4], "nps_score"] = rng.choice([-5.0, 99.0], size=len(ap[4]))

    messy = rng.choice(n, size=900, replace=False)
    df.loc[messy[:300], "primary_channel"] = "  mobile_app "
    df.loc[messy[300:600], "primary_channel"] = "WEB"
    df.loc[messy[600:], "membership_tier"] = " gold "

    _inject_missing(df, ["nps_score", "email_open_rate", "avg_shipping_delay_days", "loyalty_points", "favorite_category"], 0.032, rng)
    return _append_duplicates(df, 300, rng)


def generate_biotech_materials_wide(seed: int = 606) -> pd.DataFrame:
    """Wide high-dimensional bioinformatics & materials dataset (3,500 rows x 64 cols) with collinearity & corrupt assay wells."""
    rng = np.random.default_rng(seed)
    n = 3475  # +25 duplicates -> 3,500 rows

    # 8 latent biological/material factors driving 56 correlated descriptors (high VIF testbed)
    latent = rng.normal(0.0, 1.0, size=(n, 8))
    data: dict[str, object] = {
        "sample_well_id": [f"BIO-{20000 + i}" for i in range(n)],
        "assay_plate": rng.choice(["Plate_A1", "Plate_B2", "Plate_C3", "Plate_D4", "Plate_E5"], size=n),
        "compound_family": rng.choice(["Kinase_Inhibitor", "Peptide_Macrocycle", "Polymer_Composite", "Oxide_Catalyst"], size=n),
        "solvent_medium": rng.choice(["DMSO_01", "Aqueous_PBS", "Ethanol_Mix", "Ionic_Liquid"], size=n),
    }

    feat_mat = np.zeros((n, 58), dtype=float)
    # 20 gene-expression / spectral bands + 20 physicochemical descriptors + 18 collinear/noisy probes
    for j in range(56):
        f1 = j % 8
        f2 = (j * 3 + 1) % 8
        loading1 = rng.uniform(0.55, 1.35) * (1.0 if j % 2 == 0 else -1.0)
        loading2 = rng.uniform(0.15, 0.65)
        noise_scale = 0.12 if j >= 38 else 0.45  # Deliberate strong multicollinearity in last 18 descriptors
        col_vals = loading1 * latent[:, f1] + loading2 * latent[:, f2] + rng.normal(0, noise_scale, n)
        feat_mat[:, j] = col_vals
        prefix = "gene_expr" if j < 20 else ("mol_desc" if j < 40 else "spec_band")
        data[f"{prefix}_{j + 1:02d}"] = np.round(col_vals, 4)

    dose_um = np.round(rng.lognormal(1.8, 0.9, n).clip(0.05, 500.0), 3)
    assay_ph = np.round(rng.normal(7.2, 0.45, n).clip(4.5, 9.8), 2)
    data["dose_micromolar"] = dose_um
    data["assay_ph"] = assay_ph

    # Strongly nonlinear response (quadratic + interaction + log-dose) so neural/tree engines outperform linear baselines
    bioactivity = (
        48.0
        + 11.5 * latent[:, 0]
        - 8.2 * latent[:, 1]
        + 6.4 * (latent[:, 2] ** 2)
        + 7.8 * np.sin(1.4 * latent[:, 3])
        + 5.5 * latent[:, 0] * latent[:, 4]
        + 6.2 * np.log1p(dose_um)
        - 4.5 * (assay_ph - 7.2) ** 2
        + rng.normal(0, 3.2, n)
    )
    tox_logit = -2.2 + 0.85 * np.log1p(dose_um) + 1.1 * latent[:, 5] + 0.9 * np.abs(latent[:, 1]) - 0.7 * latent[:, 6]
    tox_prob = 1.0 / (1.0 + np.exp(-np.clip(tox_logit, -15, 15)))
    toxicity_alert = (rng.random(n) < tox_prob).astype(int)

    data["bioactivity_response"] = np.round(bioactivity, 3)
    data["toxicity_alert"] = toxicity_alert
    df = pd.DataFrame(data)

    # Inject pipetting / instrument calibration blow-up cluster (~75 wells) + extreme pH / negative dose outliers
    bad_wells = rng.choice(n, size=75, replace=False)
    for col in [f"gene_expr_{k:02d}" for k in range(1, 11)]:
        df.loc[bad_wells, col] = np.round(rng.uniform(25.0, 68.0, size=len(bad_wells)), 3)
    out_idx = rng.choice(n, size=140, replace=False)
    df.loc[out_idx[:70], "dose_micromolar"] = rng.choice([-50.0, -12.5, 9500.0], size=70)
    df.loc[out_idx[70:], "assay_ph"] = rng.choice([-2.0, 19.5, 24.0], size=70)

    _inject_missing(df, ["gene_expr_03", "mol_desc_24", "spec_band_45", "dose_micromolar", "assay_ph"], 0.04, rng)
    return _append_duplicates(df, 25, rng)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FUTURE_DIR.mkdir(parents=True, exist_ok=True)

    hc_df, hc_fut = generate_healthcare_medium()
    mfg_df, mfg_fut = generate_manufacturing_large()
    fin_df = generate_finance_large()
    eng_df = generate_energy_timeseries_large()
    ret_df = generate_retail_massive()
    bio_df = generate_biotech_materials_wide()

    outputs = [
        (OUT_DIR / "01_healthcare_clinical_readmission_medium.csv", hc_df),
        (OUT_DIR / "02_manufacturing_iot_predictive_maintenance_large.csv", mfg_df),
        (OUT_DIR / "03_finance_credit_fraud_risk_large.csv", fin_df),
        (OUT_DIR / "04_energy_smartgrid_demand_forecasting_large.csv", eng_df),
        (OUT_DIR / "05_retail_ecommerce_customer_analytics_massive.csv", ret_df),
        (OUT_DIR / "06_biotech_genomics_materials_wide_medium.csv", bio_df),
        (FUTURE_DIR / "future_healthcare_patients_incoming.csv", hc_fut),
        (FUTURE_DIR / "future_manufacturing_sensors_incoming.csv", mfg_fut),
    ]

    total_mb = 0.0
    for path, frame in outputs:
        frame.to_csv(path, index=False)
        size_mb = path.stat().st_size / (1024 * 1024)
        total_mb += size_mb
        print(f"Wrote {path.relative_to(OUT_DIR)!s:58s} | {len(frame):7,d} rows x {frame.shape[1]:2d} cols | {size_mb:6.2f} MB")
    print(f"TOTAL SIZE: {total_mb:.2f} MB across {len(outputs)} CSV files.")


if __name__ == "__main__":
    main()
