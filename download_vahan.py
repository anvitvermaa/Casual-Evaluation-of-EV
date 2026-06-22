import os
import subprocess
import shutil

states = [
    "Maharashtra", "Gujarat", "Karnataka", "Tamil Nadu", "Andhra Pradesh", 
    "Telangana", "Madhya Pradesh", "Rajasthan", "Uttar Pradesh", "Kerala", 
    "Haryana", "West Bengal", "Punjab", "Odisha", "Bihar", "Chhattisgarh"
]

out_dir = "/tmp/vahan_out"
target_dir = "data/raw/vehicle_registrations/states"

os.makedirs(target_dir, exist_ok=True)

for state in states:
    print(f"Downloading {state}...")
    cmd = [
        "/home/vityarthi/miniconda3/envs/ev-policy-sdid/bin/python",
        "/tmp/vahan-scraper/scripts/api.py",
        "--yaxis", "Vehicle Class",
        "--xaxis", "Month Wise",
        "--state", state,
        "--start-year", "2022",
        "--end-year", "2026",
        "--out", out_dir
    ]
    subprocess.run(cmd, check=True)
    
    # After download, find the state directory in out_dir.
    # The scraper uses e.g. "Maharashtra_59" or similar naming.
    state_dirs = [d for d in os.listdir(out_dir) if d.lower().startswith(state.lower().replace(" ", ""))]
    if state_dirs:
        state_dir = os.path.join(out_dir, state_dirs[0], "all_rtos", "Vehicle_Class__Month_Wise")
        if os.path.exists(state_dir):
            for file in os.listdir(state_dir):
                if file.endswith(".csv"):
                    year = file.replace(".csv", "")
                    target_file = os.path.join(target_dir, f"{state}_{year}_master.csv")
                    shutil.copy(os.path.join(state_dir, file), target_file)
                    print(f"  Copied to {target_file}")
                    
print("All done!")
