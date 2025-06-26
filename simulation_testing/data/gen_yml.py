#!/usr/bin/env python3
import os, itertools, copy, random, yaml

LAND_TEMPLATE = "./50_jun_grid_ls_master.yml"
OS_TEMPLATE   = "./50_jun_grid_os_master.yml"
OUT_PREFIX    = ""
P_VALUES      = [0.001, 0.32, 0.01, 0.032, 0.1]

def sanitise(x: float) -> str:
    return str(x).replace('.', 'p')

random.seed(123456)

with open(LAND_TEMPLATE) as fh:
    land_template = yaml.safe_load(fh)
with open(OS_TEMPLATE) as fh:
    os_template = yaml.safe_load(fh)

idx = 0
for p_pos, p_neg in itertools.product(P_VALUES, repeat=2):
    san_pos = sanitise(p_pos)
    san_neg = sanitise(p_neg)

    dir_name = f"{OUT_PREFIX}{idx:02d}_pos{san_pos}_neg{san_neg}"
    dir_ls   = os.path.join(dir_name, "ls")
    os.makedirs(dir_ls, exist_ok=True)

    # landscape.yml
    land_cfg = copy.deepcopy(land_template)
    land_cfg["simul"]["p_pos"] = p_pos
    land_cfg["simul"]["p_neg"] = p_neg
    land_cfg["rng"]["seed"]    = random.randint(0, 2**32 - 1)

    land_out_name = f"landscape_pos{san_pos}_neg{san_neg}.yml"
    land_out      = os.path.join(dir_ls, land_out_name)
    with open(land_out, "w") as fh:
        yaml.dump(land_cfg, fh, sort_keys=False)

    # os.yml
    os_cfg = copy.deepcopy(os_template)
    os_cfg["rng"]["seed"] = random.randint(0, 2**32 - 1)

    os_out_name = f"os_pos{san_pos}_neg{san_neg}.yml"
    os_out      = os.path.join(dir_name, os_out_name)
    with open(os_out, "w") as fh:
        yaml.dump(os_cfg, fh, sort_keys=False)

    print(f"Wrote {land_out} and {os_out}")
    idx += 1