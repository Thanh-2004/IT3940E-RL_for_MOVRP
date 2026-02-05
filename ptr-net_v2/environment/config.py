from __future__ import annotations
import csv, json, math, re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional

# ------------------------------
# Core config container
# ------------------------------

def _torch_available_and_has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

@dataclass(slots=True)
class Config:
    trucks: "Trucks"
    drones: "Drones"
    customers: "Customers"
    device: str = field(default_factory=lambda: "cuda" if _torch_available_and_has_cuda() else "cpu")


# ------------------------------
# Trucks
# ------------------------------

@dataclass(slots=True)
class Trucks:
    """
    Truck config example (single-fleet profile shared by all trucks):
    {
        "V_max (m/s)": 15.557,
        "T (hour)": {"0-1":0.7, "1-2":0.4, ..., "11-12":0.8}
    }
    """
    num_trucks: int
    vmax: float                       # base max speed (m/s)
    slot_coeffs: List[float]          # length-L per-hour multipliers, e.g., len=12
    slot_len_hours: float = 1.0       # slot length (hours)

    @staticmethod
    def from_json_dict(num_trucks: int, j: Dict) -> "Trucks":
        vmax = float(j["V_max (m/s)"])
        slot_map = j["T (hour)"]  # e.g. {"0-1":0.7,...}
        parsed: List[Tuple[int, float]] = []
        for k, v in slot_map.items():
            start, end = k.split("-")
            parsed.append((int(start), float(v)))
        parsed.sort(key=lambda x: x[0])
        coeffs = [c for _, c in parsed]
        if not coeffs:
            raise ValueError("Empty truck slot coefficients.")
        return Trucks(num_trucks=num_trucks, vmax=vmax, slot_coeffs=coeffs)

    @staticmethod
    def from_json_file(num_trucks: int, path: str | Path) -> "Trucks":
        with open(path, "r", encoding="utf-8") as f:
            j = json.load(f)
        return Trucks.from_json_dict(num_trucks, j)

    # ---- time-dependent speed utilities ----

    @property
    def L(self) -> int:
        return len(self.slot_coeffs)

    @property
    def slot_len_seconds(self) -> float:
        return self.slot_len_hours * 3600.0

    def speed_at(self, t_seconds: float) -> float:
        """
        v_trk(t) = vmax * sigma_a for t in slot a, cyclic over a day profile.
        """
        cycle = self.L * self.slot_len_seconds
        _, t = divmod(t_seconds, cycle)
        slot = int(t // self.slot_len_seconds)
        return self.vmax * self.slot_coeffs[slot]

    def calc_truck_travel_time(self, start_time_s: float, distance_m: float) -> float:
        """
        Integrate piecewise-constant speed starting at start_time_s.
        Returns travel time (seconds) to traverse 'distance_m'.
        """
        if distance_m <= 0:
            return 0.0

        rem = distance_m
        t = start_time_s
        total_dt = 0.0
        # Guard against pathological configs
        max_iters = 10_000 + int(distance_m / max(1.0, self.vmax))  # very loose

        it = 0
        while rem > 1e-9:
            it += 1
            if it > max_iters:
                raise RuntimeError("Integration loop exceeded safe iteration budget. Check slot coeffs.")

            v = self.speed_at(t)
            if v <= 0:
                raise ValueError("Truck speed non-positive at some slot; check coefficients.")

            # time remaining in current slot
            slot_end = math.floor(t / self.slot_len_seconds + 1) * self.slot_len_seconds
            dt_slot = max(0.0, slot_end - t)
            if dt_slot == 0.0:
                # move epsilon into next slot to avoid a stuck loop on exact boundary
                t += 1e-6
                continue

            can_travel = v * dt_slot  # meters within this slot
            if can_travel >= rem:
                total_dt += rem / v
                rem = 0.0
            else:
                rem -= can_travel
                total_dt += dt_slot
                t += dt_slot  # next slot boundary

        return total_dt  # seconds

    def __str__(self) -> str:
        preview = ", ".join(f"{c:.2f}" for c in self.slot_coeffs[:5])
        more = "" if len(self.slot_coeffs) <= 5 else f", ... (L={len(self.slot_coeffs)})"
        return (f"Trucks(K={self.num_trucks}, vmax={self.vmax:.3f} m/s, "
                f"slots=[{preview}{more}], slot_len={self.slot_len_hours}h)")


# ------------------------------
# Drone
# ------------------------------

@dataclass(slots=True, frozen=True)
class Drone:
    """
    One drone platform. Keys expected (your sample):
      "takeoffSpeed [m/s]", "cruiseSpeed [m/s]", "landingSpeed [m/s]",
      "cruiseAlt [m]", "capacity [kg]", "batteryPower [Joule]",
      "speed_type", "range", "beta(w/kg)", "gama(w)"
    """
    takeoff_speed: float
    cruise_speed: float
    landing_speed: float
    cruise_alt: float
    capacity: float
    battery_power: float
    speed_type: str
    range: str
    beta: float      # W/kg
    gama: float      # W  (spelled 'gama' in source; keep as-is)

    @staticmethod
    def from_json_dict(j: Dict) -> "Drone":
        return Drone(
            takeoff_speed=float(j["takeoffSpeed [m/s]"]),
            cruise_speed=float(j["cruiseSpeed [m/s]"]),
            landing_speed=float(j["landingSpeed [m/s]"]),
            cruise_alt=float(j["cruiseAlt [m]"]),
            capacity=float(j["capacity [kg]"]),
            battery_power=float(j["batteryPower [Joule]"]),
            speed_type=str(j.get("speed_type", "")),
            range=str(j.get("range", "")),
            beta=float(j["beta(w/kg)"]),
            gama=float(j["gama(w)"]),
        )

    # ---- kinematics/energy ----

    def flight_time_s(self, dist_m: float) -> float:
        """
        t_to + t_cruise + t_ld, with vertical segments time = alt/speed.
        """
        if dist_m < 0:
            raise ValueError("Distance must be non-negative.")
        t_to = self.cruise_alt / max(1e-9, self.takeoff_speed)
        t_cr = dist_m / max(1e-9, self.cruise_speed)
        t_ld = self.cruise_alt / max(1e-9, self.landing_speed)
        return t_to + t_cr + t_ld  # seconds

    def energy_j(self, dist_m: float, payload_kg: float) -> float:
        """
        P(w)=beta*w + gama (Watts); E = P * flight_time_s.
        """
        w = max(0.0, payload_kg)
        P = self.beta * w + self.gama
        return P * self.flight_time_s(dist_m)
    
    def pretty(self, idx: int | None = None) -> str:
        head = f"Drone[{idx}]" if idx is not None else "Drone"
        return (f"{head}(cruise={self.cruise_speed:.2f} m/s, "
                f"takeoff={self.takeoff_speed:.2f}, landing={self.landing_speed:.2f}, "
                f"alt={self.cruise_alt:.0f} m, cap={self.capacity:.2f} kg, "
                f"Emax={self.battery_power:.0f} J, beta={self.beta:.1f}, gama={self.gama:.1f})")
    
    


@dataclass(slots=True)
class Drones:
    """A fleet of drones; ids are 1..D in insertion order unless explicit keys are provided."""
    drones: List[Drone] = field(default_factory=list)
    ids: List[str] = field(default_factory=list)  # original keys from JSON
    max_trip_time_s: Optional[float] = None       # optional per-trip cap (scenario)

    @property
    def D(self) -> int:
        return len(self.drones)

    def __getitem__(self, idx: int) -> Drone:
        return self.drones[idx]

    def head(self, m: int) -> "Drones":
        """
        Shallow subset fleet of the first m drones.
        """
        m = max(0, min(m, self.D))
        return Drones(
            drones=self.drones[:m],
            ids=self.ids[:m],
            max_trip_time_s=self.max_trip_time_s
        )

    @staticmethod
    def from_json_dict(j: Dict[str, Dict]) -> "Drones":
        # keys are strings "1","2",...
        ids_sorted = sorted(j.keys(), key=lambda k: int(k) if str(k).isdigit() else k)
        drones = [Drone.from_json_dict(j[k]) for k in ids_sorted]
        return Drones(drones=drones, ids=ids_sorted)

    @staticmethod
    def from_json_file(path: str | Path) -> "Drones":
        with open(path, "r", encoding="utf-8") as f:
            j = json.load(f)
        return Drones.from_json_dict(j)

    def __str__(self) -> str:
        D = len(self.drones)
        preview = "\n\t".join(d.pretty(i) for i, d in list(enumerate(self.drones))[:2])
        more = "" if D <= 2 else f", ... (+{D-2} more)"
        cap = f", trip_cap={self.max_trip_time_s:.0f}s" if self.max_trip_time_s else ""
        return f"Drones(D={D}{cap}): [\n\t{preview}{more}\n]"


# ------------------------------
# Customers
# ------------------------------

@dataclass(slots=True, frozen=True)
class Customer:
    x: float
    y: float
    demand: float                  # kg
    only_truck: bool               # True if truck-only
    truck_service_time_s: float
    drone_service_time_s: float

    def coord(self) -> Tuple[float, float]:
        return (self.x, self.y)
    
    def pretty(self, idx: int | None = None) -> str:
        head = f"Cust[{idx}]" if idx is not None else "Cust"
        only = "truck" if self.only_truck else "any"
        return (f"{head}((x={self.x:.1f},y={self.y:.1f}), demand={self.demand:.3f} kg, "
                f"{only}, svc_trk={self.truck_service_time_s:.0f}s, "
                f"svc_dr={self.drone_service_time_s:.0f}s)")


@dataclass(slots=True)
class Customers:
    """
    CSV format (header optional if you pass column indices):
      X, Y, Demand, OnlyServicedByStaff, ServiceTimeByTruck(s), ServiceTimeByDrone(s)
    """
    items: List[Customer] = field(default_factory=list)

    @property
    def N(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Customer:
        return self.items[idx]

    @staticmethod
    def from_csv(
        path: str | Path,
        has_header: bool = True,
        col_x: int = 0,
        col_y: int = 1,
        col_demand: int = 2,
        col_only_truck: int = 3,
        col_trk_svc: int = 4,
        col_dr_svc: int = 5,
        delimiter: Optional[str] = None
    ) -> "Customers":
        records: List[Customer] = []
        with open(path, "r", encoding="utf-8") as f:
            if delimiter is None:
                sample = f.read(4096)
                f.seek(0)
                try:
                    dialect = csv.Sniffer().sniff(sample)
                    delim = dialect.delimiter
                except Exception:
                    delim = ","
            else:
                delim = delimiter

            reader = csv.reader(f, delimiter=delim)
            if has_header:
                next(reader, None)
            for row in reader:
                if not row or all((c.strip() == "" for c in row)):
                    continue
                x = float(row[col_x]); y = float(row[col_y])
                demand = float(row[col_demand])
                only_truck = bool(int(row[col_only_truck]))
                trk_s = float(row[col_trk_svc]); dr_s = float(row[col_dr_svc])
                records.append(Customer(x, y, demand, only_truck, trk_s, dr_s))
        if not records:
            raise ValueError("No customer rows parsed from CSV.")
        return Customers(records)

    @staticmethod
    def from_list(rows: Iterable[Tuple[float, float, float, int, float, float]]) -> "Customers":
        return Customers([Customer(x, y, d, bool(ot), ts, ds) for (x, y, d, ot, ts, ds) in rows])

    def __str__(self) -> str:
        N = len(self.items)
        preview = "\n\t".join(c.pretty(i) for i, c in list(enumerate(self.items))[:3])
        more = "" if N <= 3 else f", ... (+{N-3} more)"
        return f"Customers(N={N}): [\n\t{preview}{more}\n]"


# ------------------------------
# Example helpers
# ------------------------------

def build_config_from_in_memory(
    truck_num: int,
    truck_json: Dict,
    drones_json: Dict[str, Dict],
    customers_rows: Iterable[Tuple[float, float, float, int, float, float]],
) -> Config:
    trucks = Trucks.from_json_dict(truck_num, truck_json)
    drones = Drones.from_json_dict(drones_json)
    customers = Customers.from_list(customers_rows)
    return Config(trucks=trucks, drones=drones, customers=customers)


def euclid(p: Tuple[float, float], q: Tuple[float, float]) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


# ------------------------------
# Optional: precompute drone/time matrices for your MDP
# ------------------------------

def precompute_drone_times(
    drones: Drones,
    coords: List[Tuple[float, float]],   # index 0 = depot, 1..N = customers
    enforce_trip_cap_with_return: bool = False,
) -> List[List[List[float]]]:
    """
    Returns:
      T_drone[d][i][j]  : seconds
    """
    D = drones.D
    V = len(coords)
    T_dr = [[[0.0 for _ in range(V)] for _ in range(V)] for _ in range(D)]
    cap = drones.max_trip_time_s

    for dd, drone in enumerate(drones.drones):
        for i in range(V):
            for j in range(V):
                if i == j: 
                    continue
                dist_ij = euclid(coords[i], coords[j])
                t_ij = drone.flight_time_s(dist_ij)

                if enforce_trip_cap_with_return and cap is not None:
                    dist_j0 = euclid(coords[j], coords[0])
                    t_back = drone.flight_time_s(dist_j0)
                    if (t_ij + t_back) > cap:
                        T_dr[dd][i][j] = float("inf")
                        continue

                T_dr[dd][i][j] = t_ij
    return T_dr


# ------------------------------
# Scenario TXT loader (preamble + table)
# ------------------------------

@dataclass(slots=True, frozen=True)
class ScenarioMeta:
    number_staff: int
    number_drone: int
    drone_limit_flight_time_s: float

_HEADER_RE = re.compile(r"^\s*Coordinate\s+X", flags=re.IGNORECASE)

def load_customers_with_meta_from_txt(path: str | Path) -> Tuple[Customers, ScenarioMeta]:
    """
    TXT format with a preamble, then a header line, then whitespace-separated rows:
      number_staff 2
      number_drone 2
      droneLimitationFightTime(s) 3600
      Customers 20
      Coordinate X   Coordinate Y   Demand   OnlyServicedByStaff   ServiceTimeByTruck(s)  ServiceTimeByDrone(s)
      ...
    """
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
                    key, val = parts[0], parts[1]
                    preamble[key] = val
                continue

            # header already seen → parse data row (whitespace separated)
            parts = line.split()
            if len(parts) < 6:
                raise ValueError(f"Bad customer row: {line}")
            x = float(parts[0]); y = float(parts[1])
            demand = float(parts[2])
            only_truck = bool(int(parts[3]))
            trk_s = float(parts[4]); dr_s = float(parts[5])
            rows.append((x, y, demand, int(only_truck), trk_s, dr_s))

    if not rows:
        raise ValueError("No customer records parsed from TXT.")

    customers = Customers.from_list(rows)
    meta = ScenarioMeta(
        number_staff=int(preamble.get("number_staff", 1)),
        number_drone=int(preamble.get("number_drone", 1)),
        drone_limit_flight_time_s=float(preamble.get("droneLimitationFightTime(s)", 3600)),
    )
    return customers, meta


def build_config_from_files(
    truck_json_path: str | Path,
    drone_json_path: str | Path,
    customers_txt_path: str | Path,
) -> Tuple[Config, ScenarioMeta]:
    # trucks
    with open(truck_json_path, "r", encoding="utf-8") as f:
        truck_j = json.load(f)
    # customers + meta (gives number_staff/number_drone/drone limit)
    customers, meta = load_customers_with_meta_from_txt(customers_txt_path)
    # drones catalog
    with open(drone_json_path, "r", encoding="utf-8") as f:
        drones_j = json.load(f)

    trucks = Trucks.from_json_dict(num_trucks=int(meta.number_staff), j=truck_j)
    drones_all = Drones.from_json_dict(drones_j)
    drones = drones_all.head(int(meta.number_drone))
    drones.max_trip_time_s = float(meta.drone_limit_flight_time_s)

    cfg = Config(trucks=trucks, drones=drones, customers=customers)
    return cfg, meta

if __name__ == "__main__":
    cfg, meta = build_config_from_files(
        truck_json_path="/home/saphyiera/HUST/WADRL_PVRP/data/Truck_config.json",
        drone_json_path="/home/saphyiera/HUST/WADRL_PVRP/data/drone_linear_config.json",
        customers_txt_path="/home/saphyiera/HUST/WADRL_PVRP/data/random_data/200.40.2.txt")
    print(f"Customers: {cfg.customers} \n\nDrones: {cfg.drones} \n\nTrucks: {cfg.trucks} \n\nDevice: {cfg.device}")