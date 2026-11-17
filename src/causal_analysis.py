import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

SERVICE_MAP = {
    "Haircut - Men/Women": "Haircut & Styling",
    "Signature Cut & Style": "Haircut & Styling",
    "Basic Trim & Blowdry": "Haircut & Styling",
    "L'Oreal Hair Spa": "Hair Spa & Treatment",
    "Hair Spa - Basic": "Hair Spa & Treatment",
    "Spa (Hair) Nourish": "Hair Spa & Treatment",
    "Gold Radiance Facial": "Facial & Skin Care",
    "D-Tan & Cleanse": "Facial & Skin Care",
    "Hydra Skin Therapy": "Facial & Skin Care",
    "HD Bridal Package": "Bridal & Makeup",
    "Party Makeup & Drape": "Bridal & Makeup",
    "Bridal Glow Ritual": "Bridal & Makeup",
    "Aroma Full Body Spa": "Body Spa & Massage",
    "Swedish Deep Tissue": "Body Spa & Massage",
    "Ayurvedic Massage": "Body Spa & Massage",
    "Spa Mani-Pedi Combo": "Manicure & Pedicure",
    "Classic Pedicure": "Manicure & Pedicure",
    "Nail Care Ritual": "Manicure & Pedicure",
}

def clean_records(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["clean_service_category"] = df["raw_service_name"].map(SERVICE_MAP)
    df["is_walkin"] = df["client_id"].isna().astype(int)
    df["clean_no_show"] = ((df["raw_status"] == "Cancelled") & (df["cancel_lead_hours"] <= 2.0)).astype(int)
    df["clean_adv_cancel"] = ((df["raw_status"] == "Cancelled") & (df["cancel_lead_hours"] > 2.0)).astype(int)
    df["completed"] = (df["raw_status"] == "Completed").astype(int)
    return df

def run_power_analysis(p0=0.185, p1=0.165, alpha=0.05, power=0.80, daily_bookings=820):
    h = proportion_effectsize(p0, p1)
    n_arm = int(np.ceil(NormalIndPower().solve_power(effect_size=h, alpha=alpha, power=power, ratio=1.0)))
    return {"cohens_h": h, "n_per_arm": n_arm, "total_n": n_arm * 2, "days": (n_arm * 2) / daily_bookings}

def run_did_analysis(df: pd.DataFrame):
    comp = df[df["completed"] == 1].copy()
    panel = comp.groupby(["outlet_id", "region", "week_idx"])["revenue_inr"].mean().reset_index()
    panel["is_cbe"] = (panel["region"] == "Coimbatore").astype(int)
    panel["is_post"] = (panel["week_idx"] >= 0).astype(int)
    panel["did"] = panel["is_cbe"] * panel["is_post"]
    model = smf.ols("revenue_inr ~ did + C(outlet_id) + C(week_idx)", data=panel).fit(
        cov_type="cluster", cov_kwds={"groups": panel["outlet_id"]}
    )
    return model

if __name__ == "__main__":
    print("Power Analysis:", run_power_analysis())
    df_raw = pd.read_csv("data/salon_records_300k.csv")
    df_clean = clean_records(df_raw)
    did_res = run_did_analysis(df_clean)
    print("DiD Causal Lift (INR/visit):", round(did_res.params["did"], 2))
    print("95% CI:", did_res.conf_int().loc["did"].values.round(2))
