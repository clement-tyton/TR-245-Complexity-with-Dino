# %% Multi-site runner — calls run_site per site, split across both GPUs.
# Interpreter: this project's venv.  Equivalent CLI:  python run_sites.py --all --gpus 0,1
import dino_blob as db

sites = db.list_sites()                       # all sites on disk, largest-first
print(f"{len(sites)} sites")

# %% SMOKE TEST first — a few small sites, full embed (~minutes)
db.run_all_sites(sites[-4:], gpus=(0, 1))

# %% FULL RUN — every site across both GPUs  (LONG: ~10 days; uncomment when ready)
# db.run_all_sites(sites, gpus=(0, 1))

# %% Or a single site with the live per-box bar
# db.run_site(sites[0])
