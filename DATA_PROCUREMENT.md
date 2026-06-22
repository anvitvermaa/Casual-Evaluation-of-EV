# Data Procurement Guide — N=16 Expanded Panel
## Maharashtra EV Policy 2025 — SDiD Evaluation

---

## WHAT YOU ALREADY HAVE (N=9 States ✅)

| State | Files Present |
|-------|--------------|
| Maharashtra (Treated) | 2022–2026 ✅ |
| Karnataka | 2022–2026 ✅ |
| Gujarat | 2022–2026 ✅ |
| Tamil Nadu | 2022–2026 ✅ |
| Andhra Pradesh | 2022–2026 ✅ |
| Telangana | 2022–2026 ✅ |
| Madhya Pradesh | 2022–2026 ✅ |
| Rajasthan | 2022–2026 ✅ |
| Uttar Pradesh | 2022–2026 ✅ |

**Existing file format:**
- One Excel file per state per year (e.g., `maharashtra_2022.xlsx`)
- Rows = Fuel types (ELECTRIC, DIESEL, PETROL, CNG, HYBRID...)
- Columns = Months (JAN, FEB, ... DEC) + TOTAL
- Data = **Total vehicle registrations count** for that fuel type × month

---

## WHAT YOU NEED TO DOWNLOAD (7 New States)

### Source: VAHAN Dashboard
**URL:** https://vahan.parivahan.gov.in/vahanDashboard/

### Step-by-Step Download Instructions

1. Go to https://vahan.parivahan.gov.in/vahanDashboard/
2. Select: **"Fuel Wise" → "Month Wise"** report
3. For each state + year combination below, set:
   - **State Filter:** [State Name]
   - **Year Filter:** [Year]
4. Click **Download/Export as Excel**
5. Save using the naming convention: `{statename}_{year}.xlsx`

---

## THE 7 FILES PER STATE × 5 YEARS = 35 NEW FILES

| # | State | Files to Download | Border w/ MH? | Role |
|---|-------|-------------------|----------------|------|
| 1 | **Kerala** | `kerala_2022.xlsx` to `kerala_2026.xlsx` | No | Main + Donut |
| 2 | **Haryana** | `haryana_2022.xlsx` to `haryana_2026.xlsx` | No | Main + Donut |
| 3 | **West Bengal** | `westbengal_2022.xlsx` to `westbengal_2026.xlsx` | No | Main + Donut |
| 4 | **Punjab** | `punjab_2022.xlsx` to `punjab_2026.xlsx` | No | Main + Donut |
| 5 | **Odisha** | `odisha_2022.xlsx` to `odisha_2026.xlsx` | No | Main + Donut |
| 6 | **Bihar** | `bihar_2022.xlsx` to `bihar_2026.xlsx` | No | Main + Donut |
| 7 | **Chhattisgarh** | `chhattisgarh_2022.xlsx` to `chhattisgarh_2026.xlsx` | **YES** | Main only (excluded from Donut) |

> **NOTE for Donut Hole model:** Already-existing states Gujarat, Madhya Pradesh, Telangana, Karnataka are also excluded. So Donut Pool = Kerala, Haryana, West Bengal, Punjab, Odisha, Bihar, Andhra Pradesh, Rajasthan, Uttar Pradesh, Tamil Nadu → **N₀ = 10 donors**

---

## EXACT COLUMN STRUCTURE TO VERIFY

When you open each downloaded Excel, confirm it matches this pattern:

| Row | What it Contains |
|-----|-----------------|
| Row 0 | State name header (ignore) |
| Row 2 | Month headers: `JAN, FEB, MAR, APR, MAY, JUN, JUL, AUG, SEP, OCT, NOV, DEC` |
| Row 3+ | Fuel type rows: `CNG ONLY`, `DIESEL`, `ELECTRIC`, `PETROL`, `HYBRID`, etc. |
| Last row | `TOTAL` (sum across all fuel types = total registrations that month) |

**The two rows you need from each file:**
1. The **`ELECTRIC`** row → gives monthly EV registrations
2. The **`TOTAL`** row → gives monthly total registrations
3. Penetration rate = `ELECTRIC / TOTAL × 100`

---

## COVARIATES TO COLLECT (Static, One Row Per State)

These go directly into the Python model as covariate columns.
Source: **RBI Handbook of Statistics on Indian States 2023-24**
URL: https://rbidocs.rbi.org.in/rdocs/Publications/DOCs/HBS300124.xlsx

| State | GSDP per Capita (₹, FY2023-24) | Urbanisation Rate (%) |
|-------|-------------------------------|----------------------|
| Kerala | 243,000 | 47.7 |
| Haryana | 247,000 | 34.8 |
| West Bengal | 130,000 | 31.9 |
| Punjab | 185,000 | 37.5 |
| Odisha | 145,000 | 16.7 |
| Bihar | 57,000 | 11.3 |
| Chhattisgarh | 130,000 | 23.2 |

> **These are approximate values from RBI/MOSPI.** When you download the RBI handbook, use the exact figures from Table 1 (State Domestic Product) and the urbanisation column. These replace the hardcoded values in `synthetic_control.py`.

---

## OPTIONAL Q1-STRENGTHENING COVARIATES (Recommended)

If you want to push this to a top Q1 journal (Energy Policy, Transportation Research Part D), add these two additional time-varying covariates:

### 1. Public EV Charging Station Count (Monthly, by State)
- **Source:** Ministry of Power → Bureau of Energy Efficiency (BEE)
- **URL:** https://beeindia.gov.in/en/electric-vehicle-infrastructure
- **What to download:** State-wise public EVSE (Electric Vehicle Supply Equipment) count
- **Format:** Annual table → convert to monthly using forward-fill
- **Why it matters:** Removes infrastructure endogeneity — states with more chargers naturally have higher EV adoption regardless of subsidy

### 2. State-Level Average Petrol Price (Monthly)
- **Source:** PPAC (Petroleum Planning and Analysis Cell)
- **URL:** https://www.ppac.gov.in/content/149_1_PricesPetrol.aspx
- **What to download:** Monthly petrol pump price by state (₹/litre)
- **Format:** Monthly time series for each of the 16 states
- **Why it matters:** Higher petrol prices are a strong demand-side push factor for EV adoption; omitting this creates classic OVB

---

## SEGMENT-LEVEL DATA (For Heterogeneity Analysis)

For the 2W / 3W / 4W breakdown, you need a **separate VAHAN report filter:**

1. Go to VAHAN Dashboard
2. Select: **"Vehicle Class Wise" + "Fuel Wise" + "Month Wise"**
3. For each state, filter by:
   - **Vehicle Class = TWO WHEELER (NT)** → download → `{state}_{year}_2w.xlsx`
   - **Vehicle Class = THREE WHEELER (NT)** → download → `{state}_{year}_3w.xlsx`  
   - **Vehicle Class = MOTOR CAR** (or FOUR WHEELER) → download → `{state}_{year}_4w.xlsx`
4. Within each file, extract the **ELECTRIC** row and **TOTAL** row as before

> **You need this for ALL 16 states** (including the 9 you already have). Your current files are only total-vehicle aggregates, not segment-split.

---

## COMPLETE FILE CHECKLIST

### Aggregate Files (already have 9 × 5 = 45 ✅, need 7 × 5 = 35 new)

```
data/raw/vehicle_registrations/states/
├── kerala_2022.xlsx          ← DOWNLOAD
├── kerala_2023.xlsx          ← DOWNLOAD
├── kerala_2024.xlsx          ← DOWNLOAD
├── kerala_2025.xlsx          ← DOWNLOAD
├── kerala_2026.xlsx          ← DOWNLOAD
├── haryana_2022.xlsx         ← DOWNLOAD
... (same pattern for all 7 states)
└── chhattisgarh_2026.xlsx    ← DOWNLOAD
```

### Segment Files (need ALL 16 states × 5 years × 3 segments = 240 files)

```
data/raw/vehicle_registrations/segments/
├── maharashtra_2022_2w.xlsx   ← DOWNLOAD (new)
├── maharashtra_2022_3w.xlsx   ← DOWNLOAD (new)
├── maharashtra_2022_4w.xlsx   ← DOWNLOAD (new)
... (same for all 16 states × 5 years)
```

---

## SUMMARY TABLE

| Data Type | Files Needed | Source | Priority |
|-----------|-------------|--------|----------|
| Aggregate VAHAN (7 new states) | 35 xlsx | vahan.parivahan.gov.in | **CRITICAL** |
| Segment VAHAN 2W (16 states) | 80 xlsx | vahan.parivahan.gov.in | **HIGH** |
| Segment VAHAN 3W (16 states) | 80 xlsx | vahan.parivahan.gov.in | **HIGH** |
| Segment VAHAN 4W (16 states) | 80 xlsx | vahan.parivahan.gov.in | **HIGH** |
| GSDP per Capita (7 new states) | 1 table | RBI Handbook | **CRITICAL** |
| Urbanisation Rate (7 new states) | 1 table | Census/MOSPI | **CRITICAL** |
| EV Charging Stations | 1 time-series | BEE/MoP | Optional (Q1 boost) |
| Petrol Prices | 1 time-series | PPAC | Optional (Q1 boost) |

---

## ONCE DOWNLOADED: DROP FILES HERE

```
data/raw/vehicle_registrations/states/      ← aggregate xlsx files
data/raw/vehicle_registrations/segments/    ← 2w/3w/4w segment xlsx files
```

Then run: `make run-sdid` and the pipeline will auto-detect and process all new states.
