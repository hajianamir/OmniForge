# OmniForge — Multi-Discipline Benchmark CSV Datasets

> **Repository Footprint Note:** Pre-generated CSV files are intentionally omitted from this repository to keep the codebase lightweight. Run the generator script below to recreate all 6 primary datasets and 2 future-incoming batches locally on demand, or use any of OmniForge's 18 built-in instant synthetic domain generators directly in Studio 01.

This directory provides a deterministic generator for **6 realistic medium-to-massive CSV benchmark datasets** spanning 6 distinct scientific, engineering, clinical, financial, energy, and commercial disciplines — plus **2 matching future incoming CSV batches** (without target columns) for testing **Studio 06 · Future Target Prediction (File Upload)**.

Every dataset intentionally embeds realistic **data-quality hazards, extreme outliers, corrupt sensor/entry anomalies, missing values, messy string formatting, exact duplicate records, and latent cluster structure** so you can evaluate the entire 13-studio CRISP-DM workflow end-to-end.

---

## Primary Datasets

| File | Scale | Discipline | Shape (Rows × Cols) | Size | Target Column(s) | Injected Anomalies & Hazards |
|---|---|---|---:|---:|---|---|
| `01_healthcare_clinical_readmission_medium.csv` | Medium | Healthcare & Clinical Medicine | 4,500 × 18 | 0.46 MB | `readmitted_30d` (binary), `length_of_stay_days` (reg) | Negative/215-yr ages, BP > 400 mmHg, HR 0/325 bpm, glucose > 1,500 mg/dL, SpO2 999%, messy department casing/spaces, ~4.2% MAR missing, 45 exact duplicates, 4 patient risk cohorts |
| `02_manufacturing_iot_predictive_maintenance_large.csv` | Large | Industrial IoT & Mechanical Engineering | 25,000 × 22 | 3.51 MB | `equipment_failure` (binary), `remaining_useful_life_hrs` (reg), `failure_mode` (multiclass) | Sensor saturation spikes (vibration up to 999.9, temp −99°C/485°C, RPM −1500/28,500), 220-row corrupt stuck-sensor anomaly cluster (ideal for Studio 02 DBSCAN/IsolationForest removal), ~3.2% missing telemetry, 120 duplicates |
| `03_finance_credit_fraud_risk_large.csv` | Large | Quantitative Finance, Credit & Fraud | 35,000 × 20 | 4.72 MB | `default_or_fraud_flag` (imbalanced binary), `expected_loss_usd` (reg) | Heavy fat-tail Pareto income shocks ($5M–$50M+), impossible credit scores (50 / 1,250), negative loans, extreme DTI & 24h velocity fraud bursts, ~3.8% missing bureau fields, 150 duplicates |
| `04_energy_smartgrid_demand_forecasting_large.csv` | Large (Hourly TS) | Energy, Smart Grid & Climate | 40,000 × 16 | 3.89 MB | `grid_load_mw` (reg/TS), `price_spike_alert` (binary), `solar_wind_generation_mw` (reg) | 4.5+ years of hourly seasonality + nonlinear U-curve temperature load, blackout zero/negative MW metering faults, wholesale spot price surges up to $5,200/MWh, extreme weather sensor glitches (−68°C / +112°C), ~3% missing, 100 duplicates |
| `05_retail_ecommerce_customer_analytics_massive.csv` | Massive | Retail, E-Commerce & Marketing | 100,000 × 22 | 12.03 MB | `churned_90d` (binary), `customer_ltv_usd` (reg) | Automated bot accounts (600–2,500 pages/session in <0.35s), whale order outliers ($45k–$185k), negative recency days / 9,999 days, return rates > 100%, inconsistent channel/tier strings, ~3.2% missing CRM fields, 300 duplicates |
| `06_biotech_genomics_materials_wide_medium.csv` | Wide (High-Dim) | Bioinformatics, Genomics & Materials | 3,500 × 64 | 1.60 MB | `bioactivity_response` (nonlinear reg), `toxicity_alert` (binary) | 56 correlated gene/molecular/spectral descriptors with deliberate multicollinearity (high VIF), quadratic/sinusoidal/interaction target signal where Deep Learning outperforms linear baselines, 75-well instrument calibration blow-up cluster, negative dose & pH 24 outliers, ~4% missing, 25 duplicates |

---

## Matching Future Incoming Datasets (for Studio 06 · Future Target Prediction)

| File | Matches Primary Dataset | Shape | Notes |
|---|---|---:|---|
| `future_incoming/future_healthcare_patients_incoming.csv` | `01_healthcare_clinical_readmission_medium.csv` | 50 × 16 | Target columns removed; includes realistic age drift and 4 severe out-of-distribution clinical records for distribution-shift alerting. |
| `future_incoming/future_manufacturing_sensors_incoming.csv` | `02_manufacturing_iot_predictive_maintenance_large.csv` | 100 × 19 | Target columns removed; includes elevated tool wear and 6 anomalous thermal/vibration packets. |

---

## Re-generating Datasets

To regenerate all CSV files deterministically (`seed=101..606`):

```bash
python examples/datasets/generate_example_datasets.py
```
