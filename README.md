# Salon Experiment & Causal Analysis

**Author:** Ravi Dixit (`dixitravi174-cpu`)  
**Domain:** Beauty & Wellness Retail Analytics (22 Outlets, 300,000 Records)

## Overview
End-to-end experimental design and quasi-experimental causal inference pipeline evaluating operational interventions across a 22-outlet salon and spa chain:
1. **Data Engineering & Anomaly Resolution:** Reclassifies disguised no-shows (`cancel_lead_hours <= 2.0`), maps 18 branch-specific service variants into 6 canonical categories, and isolates 28% unlinked walk-in traffic.
2. **A/B Experiment Design & Power Analysis:** Designs a client-stratified experiment for a 24-hour WhatsApp reminder + INR 100 loyalty voucher (`N = 5,662/arm`, 14-day runtime at 80% power).
3. **Experiment Validation & CUPED Variance Reduction:** Verifies zero Sample Ratio Mismatch (`p = 0.8027`), covariate balance (`|SMD| < 0.03`), estimates ATE via Welch's t-test and 3,000-iteration bootstrap, and achieves **24.6% variance reduction** via CUPED (cutting required runtime from 14.0 to 10.6 days).
4. **Difference-in-Differences (DiD) & Segmentation:** Separates true causal lift (`+INR 92.63/visit`, `p < 0.0001`) from seasonal demand (`+8.1%`) across 10 Coimbatore outlets vs. 12 control outlets, validating parallel trends (`p = 0.8163`).

## Quickstart
```bash
python -m venv venv
source venv/bin/activate  # On Windows PowerShell: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/generate_data.py
python src/causal_analysis.py
```
