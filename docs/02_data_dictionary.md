# Data Dictionary & Source Registry

> **Project:** Causal Evaluation of Maharashtra EV Policy 2025  
> **Version:** 1.0 | **Date:** 2026-05-13

---

## 1. Dataset Overview

| ID | Dataset | Source Portal | Update Freq | Format | Est. Size |
|----|---------|---------------|-------------|--------|-----------|
| DS-01 | Vehicle Registration Data | OpenCity.in / Mahasdb | Monthly | CSV/Parquet | ~2–5 GB |
| DS-02 | Air Quality (PM2.5, NOx) | CPCB / Maharashtra PCB | Hourly | CSV | ~500 MB–1 GB |
| DS-03 | Economic Survey 2025-26 | Maharashtra DES | Annual | CSV/XLS | ~10 MB |
| DS-04 | EV Charging Stations | BEE / MoP / DISCOM | Quarterly | GeoJSON | ~5 MB |
| DS-05 | Highway Toll Records | MSRDC / NHAI | Daily | CSV | ~200 MB |

---

## 2. DS-01: Vehicle Registration Data

**Source:** Maharashtra State Data Bank (Mahasdb) / OpenCity.in  
**Coverage:** January 2019 – December 2025  
**Granularity:** Individual vehicle registration records, aggregatable to RTO-level monthly

### Schema

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `registration_id` | STRING | Unique vehicle registration number | MH-01-AB-1234 |
| `registration_date` | DATE | Date of registration | 2024-03-15 |
| `rto_code` | STRING | Regional Transport Office code | MH-01 |
| `rto_name` | STRING | RTO office name | Mumbai South |
| `district` | STRING | District name | Mumbai |
| `vehicle_class` | STRING | Vehicle classification | Motor Car, M-Cycle |
| `fuel_type` | STRING | Fuel type category | Electric, Petrol, Diesel, CNG |
| `maker_name` | STRING | Vehicle manufacturer | Tata Motors, Ola Electric |
| `model_name` | STRING | Vehicle model | Nexon EV, S1 Pro |
| `vehicle_category` | STRING | 2W, 3W, 4W, Bus, Commercial | 2W |
| `owner_type` | STRING | Individual / Company / Government | Individual |

### Derived Variables

| Variable | Formula | Description |
|----------|---------|-------------|
| `ev_registrations_monthly` | COUNT WHERE fuel_type='Electric' GROUP BY district, month | Monthly EV count per district |
| `total_registrations_monthly` | COUNT GROUP BY district, month | Total registrations per district |
| `ev_penetration_rate` | ev_registrations / total_registrations × 100 | EV share as % of new registrations |
| `ev_penetration_delta` | ev_pen(t) - ev_pen(t-1) | Month-over-month change |
| `ev_pen_rolling_3m` | ROLLING_AVG(ev_penetration_rate, 3) | 3-month smoothed rate |

---

## 3. DS-02: Air Quality Data (PM2.5 / NOx)

**Source:** Central Pollution Control Board (CPCB) Continuous Ambient Air Quality Monitoring  
**Coverage:** January 2019 – December 2025  
**Granularity:** Hourly readings per monitoring station

### Schema

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `station_id` | STRING | CPCB station identifier | MH_MUM_ITO_01 |
| `station_name` | STRING | Station name | Bandra Kurla Complex |
| `district` | STRING | District | Mumbai Suburban |
| `latitude` | FLOAT | Station latitude | 19.0596 |
| `longitude` | FLOAT | Station longitude | 72.8656 |
| `datetime` | TIMESTAMP | Observation timestamp (IST) | 2024-06-15 14:00:00 |
| `pm25` | FLOAT | PM2.5 concentration (μg/m³) | 45.2 |
| `pm10` | FLOAT | PM10 concentration (μg/m³) | 89.7 |
| `no2` | FLOAT | NO₂ concentration (μg/m³) | 32.1 |
| `nox` | FLOAT | NOx concentration (ppb) | 28.4 |
| `so2` | FLOAT | SO₂ concentration (μg/m³) | 8.6 |
| `co` | FLOAT | CO concentration (mg/m³) | 1.2 |

### Derived Variables

| Variable | Formula | Description |
|----------|---------|-------------|
| `pm25_daily_mean` | AVG(pm25) GROUP BY station, date | Daily average PM2.5 |
| `pm25_monthly_mean` | AVG(pm25_daily_mean) GROUP BY district, month | Monthly district aggregate |
| `pm25_corridor_mean` | AVG(pm25) WHERE station near highway corridor | Corridor-specific PM2.5 |
| `nox_monthly_mean` | AVG(nox) GROUP BY district, month | Monthly NOx per district |

---

## 4. DS-03: Economic Survey of Maharashtra 2025-26

**Source:** Directorate of Economics and Statistics (DES), Maharashtra  
**Coverage:** 2018–2025 (annual)  
**Granularity:** District-level

### Schema

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `district` | STRING | District name | Pune |
| `year` | INT | Fiscal year | 2024 |
| `gsdp_crore` | FLOAT | District GDP (₹ Crore) | 45230.5 |
| `gsdp_per_capita` | FLOAT | Per capita GSDP (₹ Lakh) | 3.21 |
| `population` | INT | Estimated population | 12,345,678 |
| `urban_population_pct` | FLOAT | Urbanization rate (%) | 68.4 |
| `literacy_rate` | FLOAT | Literacy rate (%) | 89.2 |
| `road_length_km` | FLOAT | Total road network (km) | 15,430 |
| `industrial_units` | INT | Registered industrial units | 4,521 |
| `power_consumption_mwh` | FLOAT | Annual electricity consumption | 8,930.5 |

---

## 5. DS-04: EV Charging Station Registry

**Source:** Bureau of Energy Efficiency (BEE) / State DISCOMs  
**Coverage:** 2020–2025  
**Granularity:** Individual station, point-level

### Schema

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `station_id` | STRING | Unique station identifier | MH-CS-0042 |
| `district` | STRING | District | Nagpur |
| `latitude` | FLOAT | Station latitude | 21.1458 |
| `longitude` | FLOAT | Station longitude | 79.0882 |
| `charger_type` | STRING | AC / DC Fast / DC Ultra-Fast | DC Fast |
| `power_kw` | FLOAT | Maximum charging power (kW) | 50 |
| `operator` | STRING | Charging network operator | Tata Power EZ |
| `install_date` | DATE | Installation date | 2024-08-20 |
| `vgf_funded` | BOOLEAN | Whether funded under VGF scheme | TRUE |
| `corridor_tag` | STRING | Highway corridor association | Mumbai-Pune Expressway |

### Derived Variables

| Variable | Formula | Description |
|----------|---------|-------------|
| `charging_density` | COUNT / district_area_km2 × 100 | Stations per 100 km² |
| `dc_fast_share` | COUNT(DC) / COUNT(ALL) | Share of DC fast chargers |
| `vgf_density` | COUNT(vgf_funded=TRUE) / area | VGF-funded station density |

---

## 6. Treatment Assignment Logic

### Treated Districts (Treatment Group)
Districts receiving **high exposure** to the Maharashtra EV Policy 2025 interventions:

**Criteria (ANY of the following):**
1. Located along VGF-designated EV Mobility Corridors with ≥3 VGF-funded DC fast-charging stations installed post-May 2025
2. Contains toll plazas on Mumbai-Pune Expressway or Samruddhi Mahamarg (benefiting from 100% EV toll waiver)
3. Received ≥5 e-bus subsidies under the 40% transit electrification target

**Expected treated districts:** Mumbai, Mumbai Suburban, Thane, Pune, Nashik, Nagpur, Aurangabad (Chhatrapati Sambhajinagar), Raigad

### Donor Pool (Control Group)
All remaining Maharashtra districts NOT meeting the treatment criteria above, with additional exclusions:
- Districts bordering treated districts (spatial buffer to mitigate spillover)
- Districts with population < 500,000 (insufficient registration volume for stable rates)

**Expected donor pool:** Kolhapur, Solapur, Satara, Sangli, Jalgaon, Ahmednagar, Latur, Osmanabad, Beed, Nanded, Parbhani, Hingoli, Washim, Buldhana, Akola, Amravati, Yavatmal, Wardha, Chandrapur, Gadchiroli, Gondia, Sindhudurg, Ratnagiri

---

## 7. Join Keys

All datasets are linked through the following keys:

| Primary Key | Datasets Linked | Join Type |
|-------------|-----------------|-----------|
| `district` (standardized name) | DS-01 ↔ DS-02 ↔ DS-03 ↔ DS-04 | INNER JOIN |
| `year_month` (YYYY-MM) | DS-01 ↔ DS-02 | INNER JOIN |
| `year` | DS-01 ↔ DS-03 | LEFT JOIN |

**District Name Standardization:** A lookup table (`config/district_mapping.json`) maps variant spellings (e.g., "Aurangabad" ↔ "Chhatrapati Sambhajinagar") to canonical district codes.

---

*Data dictionary last updated: 2026-05-13*
