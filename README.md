# Hospital 30-Day Readmission Risk  
### Logistic Regression + Operational Targeting Strategy (Power BI)

This project develops an end-to-end healthcare analytics workflow to predict 30-day hospital readmission risk and translate model outputs into a practical intervention strategy for hospital care management teams.

---

## Dashboard Preview
![Dashboard](assets/dashboard.png)

---

## Business Problem

Hospitals face financial penalties for excessive 30-day readmissions but cannot intervene on every discharged patient due to limited resources.

This project aims to:

- Predict the probability of patient readmission within 30 days  
- Stratify patients into actionable risk tiers  
- Simulate targeted intervention strategies (Top 10–20% highest-risk patients)

---

## Dataset

**Source:** UCI Machine Learning Repository  
**Dataset:** Diabetes 130-US Hospitals (1999–2008)

Files used:
- `diabetic_data.csv`
- `IDS_mapping.csv`

---

## Analytical Approach

- Data cleaning and feature engineering (Python)
- Baseline classification model: Logistic Regression  
  - **ROC-AUC ≈ 0.636**
- Patient risk stratification:
  - Low Risk
  - Medium Risk
  - High Risk
- Operational targeting simulation:
  - Top 10%, 15%, and 20% highest-risk patients
- Power BI dashboard for decision support

---

## Key Findings

**Overall 30-day readmission rate:** ~11.2%

### Risk Stratification Results

| Risk Tier | Readmission Rate |
|-----------|------------------|
| Low       | ~6.5%            |
| Medium    | ~10.4%           |
| High      | ~16.7%           |

### Operational Targeting Simulation

| Target Group | % of Readmissions Captured | Readmission Rate Within Group |
|-------------|-----------------------------|-------------------------------|
| Top 10%     | ~21%                        | ~23%                          |
| Top 15%     | ~28%                        | ~21%                          |
| Top 20%     | ~34%                        | ~19%                          |

This indicates that focusing on the highest-risk 10% of patients could proactively address approximately **21% of total readmissions**.

---

## Model Drivers (Logistic Regression)

Odds ratio analysis suggests that readmission risk is primarily associated with:

- Prior inpatient utilization  
- Number of diagnoses  
- Emergency department visits  
- Length of hospital stay  
- Patient age group  

These findings align with clinical expectations around patient complexity and healthcare utilization history.

---

## Repository Structure

- `scripts/`
  - 01–02: Data preparation and feature engineering  
  - 03–04: Logistic regression modeling and risk scoring  
  - 05: Capture rate analysis  
  - 06: Random forest comparison  
  - 07: Logistic regression odds ratio interpretation  

- `data/` (git-ignored): raw and processed datasets  
- `assets/`: dashboard screenshots  
- `requirements.txt`