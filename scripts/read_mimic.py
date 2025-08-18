import os
import random
import argparse
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import time

def replace(group):
    mask = group.isnull()
    group[mask] = group[~mask].mean()
    return group

def sniff_columns(path):
    """
    Read only the header of a (possibly gzipped) CSV to list its columns.
    """
    print(f"→ Sniffing columns for {os.path.basename(path)}")
    # nrows=0 reads only the header
    df = pd.read_csv(path, nrows=0)
    cols = df.columns.tolist()
    print(f"   Columns found ({len(cols)}): {cols}")
    return cols

def read_with_usecols(path, desired_cols, parse_dates=None, chunksize=None):
    """
    Read a gzipped CSV in full or in chunks, keeping only the intersection
    of desired_cols and actual columns. Warn on any missing.
    Returns a DataFrame (concatenated if chunked).
    """
    actual = sniff_columns(path)
    usecols = [c for c in desired_cols if c in actual]
    missing = set(desired_cols) - set(usecols)
    if missing:
        print(f"   ⚠️  Missing columns: {sorted(missing)} – these will be skipped.")
    if chunksize:
        parts = []
        for chunk in pd.read_csv(path,
                                 usecols=usecols,
                                 parse_dates=parse_dates,
                                 chunksize=chunksize,
                                 compression='gzip'):
            parts.append(chunk)
        df = pd.concat(parts, ignore_index=True)
    else:
        df = pd.read_csv(path,
                         usecols=usecols,
                         parse_dates=parse_dates,
                         compression='gzip')
    return df

def main(csv_dir: str):
    random.seed(22891)
    os.makedirs("data", exist_ok=True)

    # ==== 1) Small tables, load whole ====
    pat = read_with_usecols(
        os.path.join(csv_dir, "PATIENTS.csv.gz"),
        desired_cols=["SUBJECT_ID","DOB","GENDER"],
        parse_dates=["DOB"]
    )
    adm = read_with_usecols(
        os.path.join(csv_dir, "ADMISSIONS.csv.gz"),
        desired_cols=[
            "SUBJECT_ID","HADM_ID","ADMITTIME","DISCHTIME","DEATHTIME",
            "ETHNICITY","ADMISSION_TYPE","HOSPITAL_EXPIRE_FLAG",
            "HAS_CHARTEVENTS_DATA"
        ],
        parse_dates=["ADMITTIME","DISCHTIME","DEATHTIME"]
    )
    ie = read_with_usecols(
        os.path.join(csv_dir, "ICUSTAYS.csv.gz"),
        desired_cols=[
            "SUBJECT_ID","HADM_ID","ICUSTAY_ID",
            "INTIME","OUTTIME","FIRST_CAREUNIT"
        ],
        parse_dates=["INTIME","OUTTIME"]
    )



    # # Sanity columns print
    # print("→ CHARTEVENTS final columns:", ce.columns.tolist())
    # print("→ LABEVENTS   final columns:", le.columns.tolist())

    # ==== 3) Build 'den' demographics+admission+icu table ====
    # ==== build ‘den’ demographics+admission+icu table ====

    # ie, adm, pat already loaded via read_with_usecols
    # only keep those with chartevents data
    den = (
        ie.merge(adm, on=["SUBJECT_ID","HADM_ID"], how="inner")
        .merge(pat, on="SUBJECT_ID",    how="inner")
    )
    den = den[den.get("HAS_CHARTEVENTS_DATA", 1) == 1]

    # --- compute hospital LOS in days, at day‐precision ---
    admit_day  = den["ADMITTIME"].values.astype("datetime64[D]")
    discharge_day = den["DISCHTIME"].values.astype("datetime64[D]")
    den["los_hospital"] = (discharge_day - admit_day).astype(int)

    # --- compute age at admission in years (days/365) ---
    dob_day    = den["DOB"].values.astype("datetime64[D]")
    den["age"] = ((admit_day - dob_day).astype(int)) / 365.0

    # --- compute ICU LOS in hours, at hour‐precision ---
    intime_hr  = den["INTIME"].values.astype("datetime64[h]")
    outtime_hr = den["OUTTIME"].values.astype("datetime64[h]")
    den["los_icu_hr"] = (outtime_hr - intime_hr).astype(int)


 
    # ICU mortality flag
    den["mort_icu"] = (
        den["DEATHTIME"].between(den["INTIME"], den["OUTTIME"])
    ).astype(int)

    # hospital‐stay sequence and first stay
    den["hospstay_seq"] = den.groupby("SUBJECT_ID")["ADMITTIME"]\
                            .rank(method="dense").astype(int)
    den["first_hosp_stay"] = (den["hospstay_seq"] == 1).astype(int)

    # ICU‐stay sequence and first ICU stay
    den["icustay_seq"] = den.groupby("HADM_ID")["INTIME"]\
                            .rank(method="dense").astype(int)
    den["first_icu_stay"] = (den["icustay_seq"] == 1).astype(int)

    # death within 168 hours of ICU admission
    den["mort_week"] = (
        den["DEATHTIME"] <= den["INTIME"] + pd.Timedelta(hours=168)
    ).astype(int)

    # filter out short ICU stays (<48 h) and implausible ages
    # ICU stay length in hours:
    # den["los_icu_hr"] = (den["OUTTIME"] - den["INTIME"]) / np.timedelta64(1, "h")

    den = den[den["los_icu_hr"] >= 48]
    den = den[den["age"] < 300]

    # encode adult ICU, gender
    den["adult_icu"] = (~den["FIRST_CAREUNIT"].isin(["PICU","NICU"])).astype(int)
    den["gender"]    = (den["GENDER"] == "M").astype(int)

    # normalize ethnicity to one of white/black/hispanic/asian/other
    den["ethnicity"] = den["ETHNICITY"].str.lower()
    den.loc[den.ethnicity.str.contains("^white"),     "ethnicity"] = "white"
    den.loc[den.ethnicity.str.contains("^black"),     "ethnicity"] = "black"
    den.loc[den.ethnicity.str.contains("^(hisp|latin)"), "ethnicity"] = "hispanic"
    den.loc[den.ethnicity.str.contains("^asia"),      "ethnicity"] = "asian"
    den.loc[~den.ethnicity.isin(["white","black","hispanic","asian"]),
            "ethnicity"] = "other"

    # drop intermediate columns we no longer need
    den.drop(columns=[
        "HOSPITAL_EXPIRE_FLAG", "HAS_CHARTEVENTS_DATA",
        # "ROW_ID",      # if present
        "los_hospital","los_icu_hr",
        "hospstay_seq","icustay_seq",
        "ADMITTIME","DISCHTIME","INTIME","OUTTIME","DOB",
        "FIRST_CAREUNIT","GENDER","ETHNICITY"
    ], inplace=True)

    den.to_csv("data/mimic3/den.csv", index=False)

    
    # # ==== Large tables, stream in chunks ====
    start = time.perf_counter()
    ce = read_with_usecols(
        os.path.join(csv_dir, "CHARTEVENTS.csv.gz"),
        desired_cols=[
            "SUBJECT_ID","HADM_ID","ICUSTAY_ID",
            "ITEMID","VALUENUM","CHARTTIME","ERROR"
        ],
        parse_dates=["CHARTTIME"],
        chunksize=1_000_000
    )

    elapsed = time.perf_counter() - start
    print(f"⏱ Loaded CHARTEVENTS in {elapsed:.2f} seconds")     

    # ==== 4) Build vitals-first-48h ====

    # —— 48 hour vitals extraction —— #
    # Merge CHARTEVENTS with icustays to get INTIME
    ce48 = (
        ce[ce["ERROR"].fillna(0) != 1]  # exclude error rows
        .merge(
            ie[["SUBJECT_ID","HADM_ID","ICUSTAY_ID","INTIME"]],
            on=["SUBJECT_ID","HADM_ID","ICUSTAY_ID"],
            how="inner"
        )
    )

    # Restrict to first 48 hours
    ce48 = ce48[
        (ce48["CHARTTIME"] >= ce48["INTIME"]) &
        (ce48["CHARTTIME"] <= ce48["INTIME"] + pd.Timedelta(hours=48))
    ]

    # Define ItemID groups
    hr_ids    = [211, 220045]
    sysbp_ids = [51,442,455,6701,220179,220050]
    diasbp_ids= [8368,8440,8441,8555,220180,220051]
    meanbp_ids= [456,52,6702,443,220052,220181,225312]
    rr_ids    = [615,618,220210,224690]
    temp_f_ids= [223761,678]
    temp_c_ids= [223762,676]
    spo2_ids  = [646,220277]
    glu_ids   = [807,811,1529,3745,3744,225664,220621,226537]

    # Build boolean conditions
    conds = [
        ce48["ITEMID"].isin(hr_ids)    & ce48["VALUENUM"].between(0,300),
        ce48["ITEMID"].isin(sysbp_ids)& ce48["VALUENUM"].between(0,400),
        ce48["ITEMID"].isin(diasbp_ids)& ce48["VALUENUM"].between(0,300),
        ce48["ITEMID"].isin(meanbp_ids)& ce48["VALUENUM"].between(0,300),
        ce48["ITEMID"].isin(rr_ids)    & ce48["VALUENUM"].between(0,70),
        ce48["ITEMID"].isin(temp_f_ids)& ce48["VALUENUM"].between(70,120),
        ce48["ITEMID"].isin(temp_c_ids)& ce48["VALUENUM"].between(10,50),
        ce48["ITEMID"].isin(spo2_ids)  & ce48["VALUENUM"].between(0,100),
        ce48["ITEMID"].isin(glu_ids)   & (ce48["VALUENUM"] > 0),
    ]

    # Corresponding VitalIDs
    vital_names = [
        "HeartRate",
        "SysBP",
        "DiasBP",
        "MeanBP",
        "RespRate",
        "Temp",  # Fahrenheit range
        "Temp",  # Celsius range
        "SpO2",
        "Glucose",
    ]

    # And how to compute the cleaned VitalValue
    value_choices = [
        ce48["VALUENUM"],  # HR
        ce48["VALUENUM"],  # SysBP
        ce48["VALUENUM"],  # DiasBP
        ce48["VALUENUM"],  # MeanBP
        ce48["VALUENUM"],  # RespRate
        (ce48["VALUENUM"] - 32) / 1.8,  # TempF → °C
        ce48["VALUENUM"],                # TempC
        ce48["VALUENUM"],  # SpO2
        ce48["VALUENUM"],  # Glucose
    ]

    # Apply the mapping
    ce48["VitalID"]    = np.select(conds, vital_names,   default=np.nan)
    ce48["VitalValue"] = np.select(conds, value_choices, default=np.nan)

    # Keep only valid vitals, and rename columns to match your SQL
    vit48 = (
        ce48.dropna(subset=["VitalID"])
            .loc[:, ["SUBJECT_ID","HADM_ID","ICUSTAY_ID","VitalID","VitalValue","CHARTTIME"]]
            .rename(columns={"CHARTTIME":"VitalChartTime"})
            .sort_values(["SUBJECT_ID","HADM_ID","ICUSTAY_ID","VitalID","VitalChartTime"])
            .reset_index(drop=True)
    )

    # vit48 is now equivalent to your vitalsfirstday view
    print(vit48.head())
    vit48.to_csv("data/mimic3/vit48.csv", index=False)

    # # ==== Large tables, stream in chunks ====
    start = time.perf_counter()
    le = read_with_usecols(
        os.path.join(csv_dir, "LABEVENTS.csv.gz"),
        desired_cols=[
            "ROW_ID", 
            "SUBJECT_ID","HADM_ID",
            # "ICUSTAY_ID",
            "ITEMID","VALUENUM","CHARTTIME",
            "VALUE",
            "VALUEUOM"
        ],
        parse_dates=["CHARTTIME"],
        chunksize=1_000_000
    )

    elapsed = time.perf_counter() - start
    print(f"⏱ Loaded LABEVENTS in {elapsed:.2f} seconds")     

    # ==== 5) lab events ====
    # —— 48-hour labs extraction —— #
    # Merge LABEVENTS with ICUSTAYS to get INTIME
    le48 = (
        le
        .merge(
            ie[["SUBJECT_ID","HADM_ID","ICUSTAY_ID","INTIME"]],
            on=["SUBJECT_ID","HADM_ID"],
            how="inner"
        )
        # only non-null, positive lab values
        .loc[le["VALUENUM"].notna() & (le["VALUENUM"] > 0)]
    )

    # Restrict to first 48 hours
    le48 = le48[
        (le48["CHARTTIME"] >= le48["INTIME"]) &
        (le48["CHARTTIME"] <= le48["INTIME"] + pd.Timedelta(hours=48))
    ]

    # Map ITEMID → label
    lab_map = {
        50868: "ANION GAP", 50862: "ALBUMIN", 50882: "BICARBONATE",
        50885: "BILIRUBIN", 50912: "CREATININE", 50806: "CHLORIDE",
        50902: "CHLORIDE", 50809: "GLUCOSE", 50931: "GLUCOSE",
        50810: "HEMATOCRIT", 51221: "HEMATOCRIT", 50811: "HEMOGLOBIN",
        51222: "HEMOGLOBIN", 50813: "LACTATE", 50960: "MAGNESIUM",
        50970: "PHOSPHATE",51265: "PLATELET",50822: "POTASSIUM",
        50971: "POTASSIUM",51275: "PTT",51237: "INR",51274: "PT",
        50824: "SODIUM",50983: "SODIUM",51006: "BUN",
        51300: "WBC",51301: "WBC"
    }

    # Sanity check thresholds per ITEMID
    lab_thresholds = {
        50862: 10,     # ALBUMIN (g/dL)
        50868: 10000,  # ANION GAP
        50882: 10000,  # BICARBONATE
        50885: 150,    # BILIRUBIN mg/dL
        50806: 10000,  # CHLORIDE
        50902: 10000,  # CHLORIDE
        50912: 150,    # CREATININE
        50809: 10000,  # GLUCOSE
        50931: 10000,  # GLUCOSE
        50810: 100,    # HEMATOCRIT
        51221: 100,    # HEMATOCRIT
        50811: 50,     # HEMOGLOBIN
        51222: 50,     # HEMOGLOBIN
        50813: 50,     # LACTATE
        50960: 60,     # MAGNESIUM
        50970: 60,     # PHOSPHATE
        51265: 10000,  # PLATELET
        50822: 30,     # POTASSIUM
        50971: 30,     # POTASSIUM
        51275: 150,    # PTT
        51237: 50,     # INR
        51274: 150,    # PT
        50824: 200,    # SODIUM
        50983: 200,    # SODIUM
        51006: 300,    # BUN
        51300: 1000,   # WBC
        51301: 1000    # WBC
    }

    # Apply label mapping and threshold filtering
    le48["label"] = le48["ITEMID"].map(lab_map)

    # Filter out any rows with no label
    le48 = le48[le48["label"].notna()]

    # Sanity-check: set LabValue to NaN if above threshold
    def apply_threshold(row):
        thr = lab_thresholds.get(row["ITEMID"], np.inf)
        return row["VALUENUM"] if row["VALUENUM"] <= thr else np.nan

    le48["LabValue"] = le48.apply(apply_threshold, axis=1)

    # Drop rows where LabValue became NaN
    le48 = le48[le48["LabValue"].notna()]

    # Select and sort columns
    lab48 = (
        le48.loc[:, ["SUBJECT_ID","HADM_ID",
                     "ICUSTAY_ID",
                     "CHARTTIME","label","LabValue"]]
            .rename(columns={"CHARTTIME":"LabChartTime"})
            .sort_values(["SUBJECT_ID","HADM_ID",
                          "ICUSTAY_ID",
                          "label","LabChartTime"])
            .reset_index(drop=True)
    )

    # Now lab48 matches your SQL lab query
    print(lab48.head())

    # Optionally save out:
    lab48.to_csv("data/mimic3/labs_first48h.csv", index=False)

    # ===== 7) Final steps ====== 

    #=====combine all variables 

    den    = pd.read_csv(os.path.join("data/mimic3/", "den.csv"))
    vit48  = pd.read_csv(os.path.join("data/mimic3/", "vitals_first48h.csv"), parse_dates=["VitalChartTime"])
    lab48  = pd.read_csv(os.path.join("data/mimic3/", "labs_first48h.csv"), parse_dates=["LabChartTime"])

    mort_vital = den.merge(vit48,how = 'left',    on = ['subject_id', 'hadm_id', 'icustay_id'])
    mort_lab = den.merge(lab48,how = 'left',    on = ['subject_id', 'hadm_id', 'icustay_id'])


    # create means by age group and gender 
    mort_vital['age_group'] = pd.cut(mort_vital['age'], [-1,5,10,15,20, 25, 40,60, 80, 200], 
        labels = ['l5','5_10', '10_15', '15_20', '20_25', '25_40', '40_60',  '60_80', '80p'])
    mort_lab['age_group'] = pd.cut(mort_lab['age'], [-1,5,10,15,20, 25, 40,60, 80, 200], 
        labels = ['l5','5_10', '10_15', '15_20', '20_25', '25_40', '40_60',  '60_80', '80p'])

    
    # one missing variable 
    adult_vital = mort_vital[(mort_vital.adult_icu==1)]
    adult_lab = mort_lab[(mort_lab.adult_icu==1)]
    adult_vital.drop(columns=['adult_icu'], inplace=True)
    adult_lab.drop(columns=['adult_icu'], inplace=True)
    
    adult_vital.to_csv(os.path.join("data/mimic3/", 'adult_icu_vital.gz'), compression='gzip',  index = False)
    mort_lab.to_csv(os.path.join("data/mimic3/", 'adult_icu_lab.gz'), compression='gzip',  index = False)



if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv_dir", type=str, required=True,
                   help="path to folder containing *.csv.gz files")
    args = p.parse_args()



    # main(args.csv_dir)


    den    = pd.read_csv(os.path.join("data/mimic3/", "den.csv"))
    vit48  = pd.read_csv(os.path.join("data/mimic3/", "vit48.csv"), parse_dates=["VitalChartTime"])
    lab48  = pd.read_csv(os.path.join("data/mimic3/", "labs_first48h.csv"), parse_dates=["LabChartTime"])

    print(den.columns)
    print(vit48.columns)
    print(lab48.columns)
    mort_vital = den.merge(vit48,how = 'left',    on = ['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID'])
    mort_lab = den.merge(lab48,how = 'left',    on = ['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID'])


    # create means by age group and gender 
    mort_vital['age_group'] = pd.cut(mort_vital['age'], [-1,5,10,15,20, 25, 40,60, 80, 200], 
        labels = ['l5','5_10', '10_15', '15_20', '20_25', '25_40', '40_60',  '60_80', '80p'])
    mort_lab['age_group'] = pd.cut(mort_lab['age'], [-1,5,10,15,20, 25, 40,60, 80, 200], 
        labels = ['l5','5_10', '10_15', '15_20', '20_25', '25_40', '40_60',  '60_80', '80p'])

    
    # one missing variable 
    adult_vital = mort_vital[(mort_vital.adult_icu==1)]
    adult_lab = mort_lab[(mort_lab.adult_icu==1)]
    adult_vital.drop(columns=['adult_icu'], inplace=True)
    adult_lab.drop(columns=['adult_icu'], inplace=True)
    
    adult_vital.to_csv(os.path.join("data/mimic3/", 'adult_icu_vital.gz'), compression='gzip',  index = False)
    mort_lab.to_csv(os.path.join("data/mimic3/", 'adult_icu_lab.gz'), compression='gzip',  index = False)
