from typing import List, Tuple


def calculate_truck_time(
    truck_route: List[int],
    M: List[List[float]],
    Vmax_truck: float,
    truck_hour: List[float],
    truck_service_time: float
) -> Tuple[float, float]:
    """
    Calculate total travel time and total waiting time for a truck route.

    Parameters
    ----------
    truck_route : List[int]
        Sequence of customer IDs (excluding depot). Example: [1, 3, 5].
    M : List[List[float]]
        Distance matrix where M[i][j] gives the distance from node i to node j.
    Vmax_truck : float
        Maximum speed of the truck (e.g., in m/s).
    truck_hour : List[float]
        Hourly speed coefficient (length 12) representing variation by time of day.
    truck_service_time : float
        Service time at each customer (seconds).

    Returns
    -------
    Tuple[float, float]
        (total_time_needed, total_wait_time)
        - total_time_needed : total travel and service time for the route.
        - total_wait_time   : total accumulated waiting time for all customers.
    """
    # Extend route with depot (0) at start and end
    route = [0] + truck_route + [0]
    num_customers = len(route)
    time_needed = 0.0
    total_wait_time = 0.0
    time_stamps = []

    for i in range(num_customers - 1):
        distance = M[route[i]][route[i + 1]]
        hour = int(time_needed / 3600)

        # Adjust travel speed when crossing hourly intervals
        while distance - ((hour + 1) * 3600 - time_needed) * Vmax_truck * truck_hour[int(hour % 12)] > 0:
            distance -= ((hour + 1) * 3600 - time_needed) * Vmax_truck * truck_hour[int(hour % 12)]
            time_needed = (hour + 1) * 3600
            hour = int(time_needed / 3600)

        time_needed += distance / (Vmax_truck * truck_hour[int(hour % 12)]) + truck_service_time
        time_stamps.append(time_needed)

    # Remove final service time (after last depot)
    time_needed -= truck_service_time

    # Compute total waiting time
    for i in range(len(truck_route)):
        total_wait_time += (time_needed - time_stamps[i])

    return time_needed, total_wait_time


def calculate_drone_time(
    drone_route: List[int],
    M: List[List[float]],
    cruise_speed: float,
    takeoff_time: float,
    landing_time: float,
    drone_service_time: float
) -> Tuple[float, float]:
    """
    Calculate total travel time and total waiting time for a drone route.

    Parameters
    ----------
    drone_route : List[int]
        Sequence of customer IDs (excluding depot). Example: [1, 4, 2].
    M : List[List[float]]
        Distance matrix where M[i][j] gives the distance from node i to node j.
    cruise_speed : float
        Drone cruise speed (e.g., in m/s).
    takeoff_time : float
        Time for drone takeoff (seconds).
    landing_time : float
        Time for drone landing (seconds).
    drone_service_time : float
        Service time at each customer (seconds).

    Returns
    -------
    Tuple[float, float]
        (total_time_needed, total_wait_time)
        - total_time_needed : total flight and service time.
        - total_wait_time   : total accumulated waiting time for all customers.
    """
    route = [0] + drone_route + [0]
    num_customers = len(route)
    time_needed = 0.0
    total_wait_time = 0.0
    time_stamps = []

    for i in range(num_customers - 1):
        distance = M[route[i]][route[i + 1]]
        flight_time = distance / cruise_speed
        time_needed += takeoff_time + landing_time + flight_time + drone_service_time
        time_stamps.append(time_needed)

    # Remove final service time after returning to depot
    time_needed -= drone_service_time

    # Compute total waiting time
    for i in range(len(drone_route)):
        total_wait_time += (time_needed - time_stamps[i])

    return time_needed, total_wait_time


if __name__ == "__main__":
    # Example data
    M = [
        [0, 10, 20, 15],
        [10, 0, 25, 30],
        [20, 25, 0, 10],
        [15, 30, 10, 0]
    ]
    VmaxTruck = 10  # m/s
    TruckHour = [1.0] * 12
    truck_service_time = 300  # seconds

    cruiseSpeed = 20
    takeoffTime = 60
    landingTime = 60
    drone_service_time = 120

    truck_route = [1, 2, 3]
    drone_route = [1, 3]

    print(calculate_truck_time(truck_route, M, VmaxTruck, TruckHour, truck_service_time))
    print(calculate_drone_time(drone_route, M, cruiseSpeed, takeoffTime, landingTime, drone_service_time))

