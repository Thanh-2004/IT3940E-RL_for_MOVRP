#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

@dataclass
class TruckConfig:
    num_trucks: int
    vmax: float
    slot_coeffs: List[float]
    slot_len_hours: float = 1.0

    @property
    def slot_len_seconds(self) -> float:
        return self.slot_len_hours * 3600.0

    @property
    def L(self) -> int:
        return len(self.slot_coeffs)

    def speed_at(self, t_seconds: float) -> float:
        cycle = self.L * self.slot_len_seconds
        _, t = divmod(t_seconds, cycle)
        slot = int(t // self.slot_len_seconds)
        return self.vmax * self.slot_coeffs[slot]

    def travel_time(self, start_time_s: float, distance_m: float) -> float:
        if distance_m <= 0:
            return 0.0
        rem = distance_m
        t = start_time_s
        total_dt = 0.0
        max_iters = 10_000 + int(distance_m / max(1.0, self.vmax))
        it = 0
        while rem > 1e-9:
            it += 1
            if it > max_iters:
                raise RuntimeError("Truck travel time integration exceeded safe iteration budget.")
            v = self.speed_at(t)
            if v <= 0:
                raise ValueError("Truck speed non-positive; check coefficients.")
            slot_end = math.floor(t / self.slot_len_seconds + 1) * self.slot_len_seconds
            dt_slot = max(0.0, slot_end - t)
            if dt_slot == 0.0:
                t += 1e-6
                continue
            can_travel = v * dt_slot
            if can_travel >= rem:
                total_dt += rem / v
                rem = 0.0
            else:
                rem -= can_travel
                total_dt += dt_slot
                t += dt_slot
        return total_dt


@dataclass
class DroneConfig:
    takeoff_speed: float
    cruise_speed: float
    landing_speed: float
    cruise_alt: float
    capacity: float
    battery_power: float
    beta: float
    gama: float

    def flight_time_s(self, dist_m: float) -> float:
        if dist_m < 0:
            raise ValueError("Distance must be non-negative.")
        t_to = self.cruise_alt / max(1e-9, self.takeoff_speed)
        t_cr = dist_m / max(1e-9, self.cruise_speed)
        t_ld = self.cruise_alt / max(1e-9, self.landing_speed)
        return t_to + t_cr + t_ld

    def energy_j(self, dist_m: float, payload_kg: float) -> float:
        w = max(0.0, payload_kg)
        P = self.beta * w + self.gama
        return P * self.flight_time_s(dist_m)


@dataclass
class ScenarioMeta:
    number_staff: int
    number_drone: int
    drone_limit_flight_time_s: float


def load_truck_json(path: str | Path, num_trucks: int) -> TruckConfig:
    with open(path, "r", encoding="utf-8") as f:
        j = json.load(f)
    vmax = float(j["V_max (m/s)"])
    slot_map = j["T (hour)"]
    parsed: List[Tuple[int, float]] = []
    for k, v in slot_map.items():
        start, _ = k.split("-")
        parsed.append((int(start), float(v)))
    parsed.sort(key=lambda x: x[0])
    coeffs = [c for _, c in parsed]
    if not coeffs:
        raise ValueError("Empty truck slot coefficients.")
    return TruckConfig(num_trucks=num_trucks, vmax=vmax, slot_coeffs=coeffs)


def load_drones_json(path: str | Path, num_drone: int) -> List[DroneConfig]:
    with open(path, "r", encoding="utf-8") as f:
        j = json.load(f)
    keys = sorted(j.keys(), key=lambda x: int(x))
    drones: List[DroneConfig] = []
    for k in keys:
        d = j[k]
        drones.append(DroneConfig(
            takeoff_speed=float(d["takeoffSpeed [m/s]"]),
            cruise_speed=float(d["cruiseSpeed [m/s]"]),
            landing_speed=float(d["landingSpeed [m/s]"]),
            cruise_alt=float(d["cruiseAlt [m]"]),
            capacity=float(d["capacity [kg]"]),
            battery_power=float(d["batteryPower [Joule]"]),
            beta=float(d["beta(w/kg)"]),
            gama=float(d["gama(w)"]),
        ))
    if num_drone > len(drones):
        raise ValueError(f"Requested {num_drone} drones but only {len(drones)} in config.")
    return drones[:num_drone]


_HEADER_RE = re.compile(r"^\s*Coordinate\s+X", flags=re.IGNORECASE)


def load_customers_with_meta(path: str | Path) -> Tuple[ScenarioMeta, List[Tuple[float, float, float, int, float, float]]]:
    preamble: Dict[str, str] = {}
    rows: List[Tuple[float, float, float, int, float, float]] = []
    header_seen = False
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if not header_seen and _HEADER_RE.search(line):
                header_seen = True
                continue
            if not header_seen:
                parts = line.split()
                if len(parts) >= 2:
                    preamble[parts[0]] = parts[1]
                continue
            parts = line.split()
            if len(parts) < 6:
                raise ValueError(f"Bad customer row: {line}")
            x = float(parts[0]); y = float(parts[1])
            demand = float(parts[2])
            only_truck = int(parts[3])
            trk_s = float(parts[4]); dr_s = float(parts[5])
            rows.append((x, y, demand, only_truck, trk_s, dr_s))
    if not rows:
        raise ValueError("No customer records parsed from TXT.")
    meta = ScenarioMeta(
        number_staff=int(preamble.get("number_staff", 1)),
        number_drone=int(preamble.get("number_drone", 1)),
        drone_limit_flight_time_s=float(preamble.get("droneLimitationFightTime(s)", 3600)),
    )
    return meta, rows


# ----------------------------
# Env simulation (standalone)
# ----------------------------

@dataclass
class EnvState:
    coords: List[Tuple[float, float]]
    demands: List[float]
    only_truck: List[int]
    svc_trk: List[float]
    svc_dr: List[float]
    trucks: TruckConfig
    drones: List[DroneConfig]
    max_trips_per_drone: Optional[int]
    trip_time_cap: Optional[float]

    # dynamic
    pos_trk: List[int]
    pos_dr: List[int]
    t_trk: List[float]
    t_dr: List[float]
    trk_done: List[bool]
    served: List[bool]
    t_svc: List[float]
    t_ret: List[float]
    cargo_trk: List[List[int]]
    cargo_dr: List[List[int]]
    cargo_mass_dr: List[float]
    Ecur: List[float]
    trip_idx: List[int]
    trip_elapsed: List[float]


def init_state(meta: ScenarioMeta, trucks: TruckConfig, drones: List[DroneConfig],
               rows: List[Tuple[float, float, float, int, float, float]],
               max_trips_per_drone: Optional[int]) -> EnvState:
    coords = [(0.0, 0.0)] + [(r[0], r[1]) for r in rows]
    demands = [0.0] + [r[2] for r in rows]
    only_truck = [0] + [r[3] for r in rows]
    svc_trk = [0.0] + [r[4] for r in rows]
    svc_dr = [0.0] + [r[5] for r in rows]
    K = meta.number_staff
    D = meta.number_drone
    pos_trk = [0] * K
    pos_dr = [0] * D
    t_trk = [0.0] * K
    t_dr = [0.0] * D
    trk_done = [False] * K
    served = [False] * (len(coords))
    served[0] = True
    t_svc = [float("nan")] * len(coords)
    t_ret = [float("nan")] * len(coords)
    cargo_trk = [[] for _ in range(K)]
    cargo_dr = [[] for _ in range(D)]
    cargo_mass_dr = [0.0] * D
    Ecur = [d.battery_power for d in drones]
    trip_idx = [0] * D
    trip_elapsed = [0.0] * D
    return EnvState(
        coords=coords,
        demands=demands,
        only_truck=only_truck,
        svc_trk=svc_trk,
        svc_dr=svc_dr,
        trucks=trucks,
        drones=drones,
        max_trips_per_drone=max_trips_per_drone,
        trip_time_cap=meta.drone_limit_flight_time_s,
        pos_trk=pos_trk,
        pos_dr=pos_dr,
        t_trk=t_trk,
        t_dr=t_dr,
        trk_done=trk_done,
        served=served,
        t_svc=t_svc,
        t_ret=t_ret,
        cargo_trk=cargo_trk,
        cargo_dr=cargo_dr,
        cargo_mass_dr=cargo_mass_dr,
        Ecur=Ecur,
        trip_idx=trip_idx,
        trip_elapsed=trip_elapsed,
    )


def euclid(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.hypot(dx, dy)


def drone_time(dr: DroneConfig, i: int, j: int, coords: List[Tuple[float, float]]) -> float:
    return dr.flight_time_s(euclid(coords[i], coords[j]))


def drone_energy(dr: DroneConfig, i: int, j: int, coords: List[Tuple[float, float]], payload: float) -> float:
    return dr.energy_j(euclid(coords[i], coords[j]), payload)


def compute_F1_F2(state: EnvState) -> Tuple[float, float]:
    t_ret = list(state.t_ret)
    # truck cargo: assume immediate return
    for k in range(len(state.pos_trk)):
        if state.cargo_trk[k]:
            i = state.pos_trk[k]
            t0 = state.t_trk[k]
            dt = state.trucks.travel_time(t0, euclid(state.coords[i], state.coords[0]))
            t_back = t0 + dt
            for cust in state.cargo_trk[k]:
                t_ret[cust] = t_back
    # drone cargo: assume immediate return
    for d in range(len(state.pos_dr)):
        if state.cargo_dr[d]:
            i = state.pos_dr[d]
            t_back = state.t_dr[d] + drone_time(state.drones[d], i, 0, state.coords)
            for cust in state.cargo_dr[d]:
                t_ret[cust] = t_back
    served_mask = [s for s in state.served]
    served_mask[0] = False
    # F1
    t_ret_vals = [t_ret[i] for i, s in enumerate(served_mask) if s and not math.isnan(t_ret[i])]
    F1 = max(t_ret_vals) if t_ret_vals else 0.0
    # F2
    F2 = 0.0
    for i, s in enumerate(served_mask):
        if not s:
            continue
        if math.isnan(t_ret[i]) or math.isnan(state.t_svc[i]):
            continue
        F2 += (t_ret[i] - state.t_svc[i])
    return F1, F2


# ----------------------------
# Transition application
# ----------------------------

def apply_truck_move(state: EnvState, k: int, j: int) -> Optional[str]:
    if k < 0 or k >= len(state.pos_trk):
        return f"truck index out of range: {k}"
    if j < 0 or j >= len(state.coords):
        return f"destination out of range: {j}"
    if state.trk_done[k]:
        return f"truck {k} already done its route"
    i = state.pos_trk[k]
    if i == j:
        return f"truck {k} self-loop at {i}"
    if j != 0 and state.served[j]:
        return f"truck {k} destination {j} already served"
    t0 = state.t_trk[k]
    dt = state.trucks.travel_time(t0, euclid(state.coords[i], state.coords[j]))
    state.t_trk[k] = t0 + dt + state.svc_trk[j]
    state.pos_trk[k] = j
    if j != 0:
        state.served[j] = True
        state.t_svc[j] = state.t_trk[k]
        state.cargo_trk[k].append(j)
    elif i != 0:
        t_back = state.t_trk[k]
        for cust in state.cargo_trk[k]:
            state.t_ret[cust] = t_back
        state.cargo_trk[k].clear()
        state.trk_done[k] = True
    return None


def apply_drone_move(state: EnvState, d: int, j: int) -> Optional[str]:
    if d < 0 or d >= len(state.pos_dr):
        return f"drone index out of range: {d}"
    if j < 0 or j >= len(state.coords):
        return f"destination out of range: {j}"
    i = state.pos_dr[d]
    if i == j:
        return f"drone {d} self-loop at {i}"
    # max trips cap
    if state.max_trips_per_drone is not None:
        if state.trip_idx[d] >= state.max_trips_per_drone and i == 0 and j != 0:
            return f"drone {d} max trips reached at depot"
    dr = state.drones[d]
    payload = state.cargo_mass_dr[d]
    Ecur = state.Ecur[d]
    # return to depot
    if j == 0:
        e_i0 = drone_energy(dr, i, 0, state.coords, payload)
        t_i0 = drone_time(dr, i, 0, state.coords)
        if Ecur < e_i0:
            return f"drone {d} insufficient energy to return (E={Ecur:.1f}, need={e_i0:.1f})"
        if state.trip_time_cap is not None and (state.trip_elapsed[d] + t_i0 > state.trip_time_cap):
            return f"drone {d} trip time cap exceeded on return"
        state.t_dr[d] += t_i0
        state.trip_elapsed[d] += t_i0
        state.Ecur[d] -= e_i0
        state.pos_dr[d] = 0
        t_back = state.t_dr[d]
        for cust in state.cargo_dr[d]:
            state.t_ret[cust] = t_back
        state.cargo_dr[d].clear()
        state.cargo_mass_dr[d] = 0.0
        # reset for next trip
        state.trip_idx[d] += 1
        state.trip_elapsed[d] = 0.0
        state.Ecur[d] = dr.battery_power
        return None
    # customer
    if state.only_truck[j]:
        return f"drone {d} cannot serve truck-only customer {j}"
    if state.served[j]:
        return f"drone {d} destination {j} already served"
    if payload + state.demands[j] > dr.capacity:
        return f"drone {d} capacity exceeded for customer {j}"
    e_ij = drone_energy(dr, i, j, state.coords, payload)
    e_j0 = drone_energy(dr, j, 0, state.coords, payload + state.demands[j])
    if Ecur < e_ij + e_j0 + 1e-9:
        return f"drone {d} insufficient energy for leg+return"
    if state.trip_time_cap is not None:
        t_ij = drone_time(dr, i, j, state.coords)
        t_j0 = drone_time(dr, j, 0, state.coords)
        svc = state.svc_dr[j]
        if state.trip_elapsed[d] + t_ij + svc + t_j0 > state.trip_time_cap:
            return f"drone {d} trip time cap exceeded for customer {j}"
    # apply move
    t_ij = drone_time(dr, i, j, state.coords)
    svc = state.svc_dr[j]
    state.t_dr[d] += t_ij + svc
    state.trip_elapsed[d] += t_ij + svc
    state.Ecur[d] -= e_ij
    state.pos_dr[d] = j
    state.served[j] = True
    state.t_svc[j] = state.t_dr[d]
    state.cargo_dr[d].append(j)
    state.cargo_mass_dr[d] += state.demands[j]
    return None


# ----------------------------
# CLI
# ----------------------------

def load_transitions(path: str | Path) -> List[Tuple[int, int, int]]:
    rows: List[Tuple[int, int, int]] = []
    int_re = re.compile(r"-?\d+")
    with open(path, "r", encoding="utf-8") as f:
        for ln, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            nums = [int(x) for x in int_re.findall(line)]
            if len(nums) != 3:
                raise ValueError(f"Bad transition line {ln}: {raw.strip()}")
            rows.append((nums[0], nums[1], nums[2]))
    return rows


# def main() -> int:
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--truck_json", required=True)
#     ap.add_argument("--drone_json", required=True)
#     ap.add_argument("--customers_txt", required=True)
#     ap.add_argument("--solutions_txt", required=True)
#     ap.add_argument("--max_trips_per_drone", type=int, default=None)
#     args = ap.parse_args()

#     meta, rows = load_customers_with_meta(args.customers_txt)
#     trucks = load_truck_json(args.truck_json, meta.number_staff)
#     drones = load_drones_json(args.drone_json, meta.number_drone)

#     state = init_state(meta, trucks, drones, rows, args.max_trips_per_drone)
#     transitions = load_transitions(args.solutions_txt)

#     invalids = []
#     for step_idx, (veh_type, veh_idx, dest) in enumerate(transitions):
#         if veh_type == 0:
#             err = apply_truck_move(state, veh_idx, dest)
#         elif veh_type == 1:
#             err = apply_drone_move(state, veh_idx, dest)
#         else:
#             err = f"unknown vehicle type {veh_type}"
#         if err:
#             info = {
#                 "step": step_idx,
#                 "action": (veh_type, veh_idx, dest),
#                 "error": err,
#             }
#             if veh_type == 1 and 0 <= veh_idx < len(state.drones):
#                 dr = state.drones[veh_idx]
#                 info["drone_energy"] = f"{state.Ecur[veh_idx]:.1f}/{dr.battery_power:.1f}"
#                 info["drone_payload"] = f"{state.cargo_mass_dr[veh_idx]:.3f}/{dr.capacity:.3f}"
#             invalids.append(info)
#             break

#     if invalids:
#         print("INVALID transitions:")
#         for item in invalids:
#             print(item)
#         return 1

#     F1_s, F2_s = compute_F1_F2(state)
#     F1_h = F1_s / 3600.0
#     F2_h = F2_s / 3600.0
#     served = sum(1 for s in state.served[1:] if s)
#     unserved = (len(state.served) - 1) - served

#     print("VALID solutions")
#     print(f"Served customers: {served}, Unserved: {unserved}")
#     print(f"F1: {F1_s:.3f} seconds, {F1_h:.6f} hours")
#     print(f"F2: {F2_s:.3f} seconds, {F2_h:.6f} hours")
#     return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--truck_json", required=True)
    ap.add_argument("--drone_json", required=True)
    ap.add_argument("--customers_txt", required=True)
    # Thay đổi từ file đơn lẻ sang directory
    ap.add_argument("--solutions_dir", required=True, help="Folder containing solution .txt files")
    ap.add_argument("--max_trips_per_drone", type=int, default=None)
    args = ap.parse_args()

    # Load dữ liệu môi trường chung một lần
    meta, rows = load_customers_with_meta(args.customers_txt)
    trucks_config = load_truck_json(args.truck_json, meta.number_staff)
    drones_config = load_drones_json(args.drone_json, meta.number_drone)

    solutions_path = Path(args.solutions_dir)
    if not solutions_path.is_dir():
        print(f"Error: {args.solutions_dir} is not a directory.")
        return 1

    # Lấy danh sách các file .txt, bỏ qua evaluation_summary.txt
    solution_files = [
        f for f in solutions_path.glob("*.txt") 
        if f.name != "evaluation_summary.txt"
    ]

    if not solution_files:
        print(f"No solution files found in {args.solutions_dir}")
        return 0

    print(f"Found {len(solution_files)} solution files. Starting evaluation...\n")
    print(f"{'File Name':<40} | {'Status':<10} | {'F1 (h)':<10} | {'F2 (h)':<10}")
    print("-" * 80)

    for sol_file in sorted(solution_files):
        # Khởi tạo lại trạng thái môi trường cho mỗi file lời giải
        state = init_state(meta, trucks_config, drones_config, rows, args.max_trips_per_drone)
        
        try:
            transitions = load_transitions(sol_file)
            error_info = None

            for step_idx, (veh_type, veh_idx, dest) in enumerate(transitions):
                if veh_type == 0:
                    err = apply_truck_move(state, veh_idx, dest)
                elif veh_type == 1:
                    err = apply_drone_move(state, veh_idx, dest)
                else:
                    err = f"unknown vehicle type {veh_type}"
                
                if err:
                    error_info = {"step": step_idx, "action": (veh_type, veh_idx, dest), "error": err}
                    break

            if error_info:
                print(f"{sol_file.name[:40]:<40} | INVALID    | Step {error_info['step']}: {error_info['error']}")
            else:
                f1_s, f2_s = compute_F1_F2(state)
                print(f"{sol_file.name[:40]:<40} | VALID      | {f1_s/3600.0:<10.4f} | {f2_s/3600.0:<10.4f}")

        except Exception as e:
            print(f"{sol_file.name[:40]:<40} | ERROR      | {str(e)}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
