import os
import subprocess
import shutil
import glob

states = [
    "Maharashtra", "Gujarat", "Karnataka", "Tamil Nadu", "Andhra Pradesh",
    "Telangana", "Madhya Pradesh", "Rajasthan", "Uttar Pradesh", "Kerala",
    "Haryana", "West Bengal", "Punjab", "Odisha", "Bihar", "Chhattisgarh"
]

out_dir = "/tmp/vahan_out_fuel"
target_dir = "data/raw/vehicle_registrations/states"

os.makedirs(out_dir, exist_ok=True)
os.makedirs(target_dir, exist_ok=True)

for state in states:
    print(f"\nDownloading FUEL data: {state}...")
    cmd = [
        "/home/vityarthi/miniconda3/envs/ev-policy-sdid/bin/python",
        "/tmp/vahan-scraper/scripts/api.py",
        "--yaxis", "Fuel",
        "--xaxis", "Month Wise",
        "--state", state,
        "--start-year", "2022",
        "--end-year", "2026",
        "--out", out_dir
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout[-500:] if result.stdout else "")
    if result.returncode != 0:
        print(f"  [ERROR] {result.stderr[-200:]}")
        continue

    # Find the scraped dir — naming pattern: {State}_{N}
    state_key = state.replace(" ", "_")
    matched = glob.glob(os.path.join(out_dir, f"{state_key}*"))
    if not matched:
        print(f"  [WARN] No output dir found for {state}")
        continue

    csv_dir = os.path.join(matched[0], "all_rtos", "Fuel__Month_Wise")
    if not os.path.exists(csv_dir):
        print(f"  [WARN] Expected dir not found: {csv_dir}")
        continue

    for file in sorted(os.listdir(csv_dir)):
        if file.endswith(".csv"):
            year = file.replace(".csv", "")
            target_file = os.path.join(target_dir, f"{state}_{year}_fuel_master.csv")
            shutil.copy(os.path.join(csv_dir, file), target_file)
            print(f"  ✅ {state}_{year}_fuel_master.csv")

print("\n\n🎉 All FUEL CSVs downloaded and copied!")
