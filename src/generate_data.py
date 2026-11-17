import os
import numpy as np
import pandas as pd

def generate_salon_datasets(seed: int = 42, output_dir: str = "data"):
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    N = 300_000
    outlets = np.array([f"CBE_{i:02d}" for i in range(1, 11)] + [f"CTL_{i:02d}" for i in range(11, 23)])
    outlet_idx = np.random.randint(0, 22, size=N)
    outlet_col = outlets[outlet_idx]
    is_cbe = (outlet_idx < 10).astype(int)
    is_flagship = np.isin(outlet_idx, [0, 1, 2, 10, 11, 12, 13]).astype(int)
    weeks = np.random.randint(-12, 14, size=N)
    is_post_qtr = (weeks >= 0).astype(int)

    is_walkin = np.random.binomial(1, 0.28, size=N)
    client_ids = np.where(is_walkin == 1, None, [f"CID_{c:05d}" for c in np.random.randint(1, 45001, size=N)])

    variants = {
        "Haircut & Styling": ["Haircut - Men/Women", "Signature Cut & Style", "Basic Trim & Blowdry"],
        "Hair Spa & Treatment": ["L'Oreal Hair Spa", "Hair Spa - Basic", "Spa (Hair) Nourish"],
        "Facial & Skin Care": ["Gold Radiance Facial", "D-Tan & Cleanse", "Hydra Skin Therapy"],
        "Bridal & Makeup": ["HD Bridal Package", "Party Makeup & Drape", "Bridal Glow Ritual"],
        "Body Spa & Massage": ["Aroma Full Body Spa", "Swedish Deep Tissue", "Ayurvedic Massage"],
        "Manicure & Pedicure": ["Spa Mani-Pedi Combo", "Classic Pedicure", "Nail Care Ritual"]
    }
    cats = list(variants.keys())
    cat_idx = np.random.choice(6, size=N, p=[0.32, 0.22, 0.20, 0.06, 0.10, 0.10])
    raw_service = [np.random.choice(variants[cats[i]]) for i in cat_idx]

    base_prices = np.array([550.0, 1450.0, 1800.0, 6500.0, 2400.0, 950.0])[cat_idx]
    tier_mult = np.where(is_flagship == 1, 1.15, 0.95)
    seasonal = 1.0 + 0.085 * np.sin((weeks + 12) / 26 * np.pi) + np.where(weeks >= 0, 0.088, 0.0)
    did_lift = np.where(
        (is_cbe == 1) & (is_post_qtr == 1),
        np.where(is_walkin == 1, -0.022, np.where(np.isin(cat_idx, [1, 2, 4]), 0.095, 0.032)),
        0.0
    )

    true_no_show = np.where(is_walkin == 1, 0, np.random.binomial(1, 0.185, size=N))
    adv_cancel = np.where((is_walkin == 1) | (true_no_show == 1), 0, np.random.binomial(1, 0.116, size=N))
    raw_status = np.where((true_no_show == 1) | (adv_cancel == 1), "Cancelled", "Completed")
    cancel_lead_hours = np.where(
        true_no_show == 1, np.random.uniform(0.0, 1.95, size=N),
        np.where(adv_cancel == 1, np.random.uniform(4.0, 48.0, size=N), np.nan)
    )
    revenue = np.where(
        raw_status == "Completed",
        np.maximum(150.0, base_prices * tier_mult * seasonal * (1.0 + did_lift) * np.random.normal(1.0, 0.18, size=N)),
        0.0
    )

    df = pd.DataFrame({
        "record_id": np.arange(1, N + 1),
        "outlet_id": outlet_col,
        "region": np.where(is_cbe == 1, "Coimbatore", "Control_TN"),
        "outlet_tier": np.where(is_flagship == 1, "Flagship", "Standard"),
        "week_idx": weeks,
        "client_id": client_ids,
        "raw_service_name": raw_service,
        "raw_status": raw_status,
        "cancel_lead_hours": np.round(cancel_lead_hours, 2),
        "revenue_inr": np.round(revenue, 2)
    })
    df.to_csv(os.path.join(output_dir, "salon_records_300k.csv"), index=False)
    print(f"Saved {len(df):,} records to {output_dir}/salon_records_300k.csv")

if __name__ == "__main__":
    generate_salon_datasets()
