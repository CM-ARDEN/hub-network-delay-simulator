import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_synthetic_network_data(
    num_units: int = 24,
    num_nodes: int = 12,
    sim_days: int = 14,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generates 100% synthetic hub-and-spoke network operations data
    using Poisson arrival shocks and Weibull recovery distributions.
    """
    rng = np.random.default_rng(seed)

    hub = "HUB-ALPHA"
    spokes = [f"NODE-{i:02d}" for i in range(1, num_nodes + 1)]
    unit_classes = ["CLASS-S", "CLASS-M", "CLASS-L"]
    min_turnaround_map = {"CLASS-S": 35, "CLASS-M": 45, "CLASS-L": 60}

    units = [
        {
            "unit_id": f"UNIT-A{i:02d}",
            "unit_class": rng.choice(unit_classes, p=[0.3, 0.5, 0.2]),
        }
        for i in range(1, num_units + 1)
    ]

    disruption_causes = [
        "NONE",
        "WEATHER_CORRIDOR",
        "CAPACITY_CONGESTION",
        "TECHNICAL_INSPECTION",
        "RESOURCE_TURNAROUND",
    ]
    cause_probs = [0.76, 0.08, 0.07, 0.05, 0.04]

    records = []
    op_counter = 1000
    base_date = datetime(2026, 1, 1, 6, 0)

    for day in range(sim_days):
        current_day_start = base_date + timedelta(days=day)

        for unit in units:
            u_id = unit["unit_id"]
            u_class = unit["unit_class"]
            min_turnaround = min_turnaround_map[u_class]

            # Each unit performs 4 to 6 chained legs per day (Hub <-> Spoke rotations)
            legs_today = int(rng.integers(4, 7))
            current_time = current_day_start + timedelta(minutes=int(rng.integers(0, 90)))
            current_station = hub if rng.random() < 0.75 else rng.choice(spokes)
            accumulated_propagated_delay = 0.0

            for leg_seq in range(1, legs_today + 1):
                op_counter += 1
                op_id = f"OP-{op_counter}"

                if current_station == hub:
                    destination = str(rng.choice(spokes))
                else:
                    destination = hub

                transit_duration = int(rng.integers(55, 195))
                scheduled_dep = current_time
                scheduled_arr = scheduled_dep + timedelta(minutes=transit_duration)

                # Primary shock generation (Poisson-triggered Weibull magnitude)
                cause = str(rng.choice(disruption_causes, p=cause_probs))
                if cause == "NONE":
                    primary_delay = 0.0
                else:
                    primary_delay = float(np.round(rng.weibull(1.4) * 28.0 + 5.0, 1))

                # Total departure delay = propagated delay from previous leg + new primary shock
                total_dep_delay = round(accumulated_propagated_delay + primary_delay, 1)

                # Transit absorption (slight recovery possible during transit)
                transit_recovery = float(np.round(rng.uniform(0.0, 5.0), 1)) if total_dep_delay > 0 else 0.0
                total_arr_delay = max(0.0, round(total_dep_delay - transit_recovery, 1))

                actual_dep = scheduled_dep + timedelta(minutes=total_dep_delay)
                actual_arr = scheduled_arr + timedelta(minutes=total_arr_delay)

                # Scheduled buffer before next leg
                scheduled_buffer = int(rng.integers(min_turnaround, min_turnaround + 40))
                absorbable_slack = max(0, scheduled_buffer - min_turnaround)

                # Propagated delay passed to the next sequential operation
                next_propagated = max(0.0, round(total_arr_delay - absorbable_slack, 1))

                records.append(
                    {
                        "sim_date": current_day_start.strftime("%Y-%m-%d"),
                        "op_id": op_id,
                        "unit_id": u_id,
                        "unit_class": u_class,
                        "leg_sequence": leg_seq,
                        "origin": current_station,
                        "destination": destination,
                        "scheduled_dep": scheduled_dep,
                        "scheduled_arr": scheduled_arr,
                        "actual_dep": actual_dep,
                        "actual_arr": actual_arr,
                        "min_turnaround_min": min_turnaround,
                        "scheduled_buffer_min": scheduled_buffer,
                        "absorbable_slack_min": absorbable_slack,
                        "disruption_cause": cause,
                        "primary_delay_min": primary_delay,
                        "propagated_delay_in_min": accumulated_propagated_delay,
                        "total_dep_delay_min": total_dep_delay,
                        "total_arr_delay_min": total_arr_delay,
                        "propagated_delay_out_min": next_propagated,
                        "is_delayed_15": int(total_arr_delay >= 15.0),
                    }
                )

                # Advance unit state for next leg
                accumulated_propagated_delay = next_propagated
                current_station = destination
                current_time = scheduled_arr + timedelta(minutes=scheduled_buffer)

    return pd.DataFrame(records)