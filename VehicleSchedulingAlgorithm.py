import numpy as np
import pandas as pd
import sys


class VSA():
    """
    Vehicle Scheduling Algorithm
    """
    def __init__(self, distance_matrix, demand_matrix, max_relocation_distance=None):
        self.distance_matrix = distance_matrix
        self.demand_matrix = demand_matrix
        self.max_relocation_distance = max_relocation_distance
        self.nr_zones = self.demand_matrix.shape[0]
        self.max_n_intervals = self.demand_matrix.shape[2]

    def solve(self):
        # set useful variables
        if self.max_relocation_distance is None:
            self.max_relocation_distance = np.max(self.distance_matrix)
        self.max_relocation_distance = min(self.max_relocation_distance, self.max_n_intervals)
        self.available_vehicles = np.zeros(shape=(self.nr_zones, self.max_n_intervals))
        self.vehicle_matrix = self.demand_matrix.copy()  # (a, b, t) -> these are the vehicles LEAVING zone a at time t towards zone b
        # run
        self._find_tours()
        # compute simple kpis
        self.vehicles_at_start = sum(self.vehicle_matrix[:, :, 0].T)
        self.required_vehicles = self.vehicles_at_start.sum()

    def to_dataframes(self):
        for attr in ['vehicle_matrix', 'demand_matrix']:  # 3D matrices
            if getattr(self, attr) is None:
                raise ValueError(f'Attribute {attr} is not set. Call solve() first.')
            array = getattr(self, attr)
            df = pd.DataFrame(array.reshape(array.shape[0] * array.shape[1], array.shape[2]))
            zone_index = np.arange(array.shape[0])
            df.index = pd.MultiIndex.from_product([zone_index, zone_index], names=['from', 'to'])
            # set attribute
            setattr(self, attr + '_df', df)

        for attr in ['distance_matrix']:  # 2D matrices
            array = getattr(self, attr)
            df = pd.DataFrame(array.reshape(array.shape[0] * array.shape[1], 1))
            zone_index = np.arange(array.shape[0])
            df.index = pd.MultiIndex.from_product([zone_index, zone_index], names=['from', 'to'])
            # set attribute
            setattr(self, attr + '_df', df)

        for attr in ['vehicles_at_start']:  # 1D matrices
            array = getattr(self, attr)
            setattr(self, attr + '_df', pd.DataFrame(array))

    def _compute_neighbours(self):
        # Fill maps with neighbouring zones for each distance:
        # zoneId -> (distance -> zones) */
        self.all_neighbours_to = {}
        for s in range(self.nr_zones):
            neighbours_to_S = dict()
            for t in range(self.nr_zones):
                if t != s:
                    if (self.distance_matrix[t, s]) not in neighbours_to_S:	 # s,t zu t,s geaendert
                        neighbours_to_S[(self.distance_matrix[t, s])] = []
                    neighbours_to_S.get((self.distance_matrix[t, s])).append(t)
            self.all_neighbours_to.update({s: neighbours_to_S})

    def _find_tours(self):
        # Fill maps with neighbouring zones for each distance:
        # zoneId -> (distance -> zones) */
        self._compute_neighbours()
        # ALGORITHM 1: go through network and set the flow to accommodate the demand
        for i in range(self.max_n_intervals):
            # Consider zone by zone
            for s in range(self.nr_zones):  
                # Consider each link in network and its demand
                for t in range(self.nr_zones):
                    # Add the routed vehicles to the available vehicles
                    # at their destination (only if still in range)
                    if (i + (self.distance_matrix[s, t])) < self.max_n_intervals:
                        self.available_vehicles[t, i + self.distance_matrix[s, t]] += self.vehicle_matrix[s, t, i]

                # Check where to get the needed vehicles from
                # (same zone, other zones or acquisition)
                self._consider_vertex(s, i)

                # If there are still vehicles available, let them stay in the zone
                if (self.available_vehicles[s, i] > 0.0):
                    self.vehicle_matrix[s, s, i] += self.available_vehicles[s, i]
                    if ((i + 1) < self.max_n_intervals):
                        self.available_vehicles[s, (i + 1)] += self.available_vehicles[s, i]

    def _consider_vertex(self, s, i):
        # Compute the total demand at zone s in time interval i
        demand_si = self.demand_matrix[s, :, i].sum()
        # check whether enough vehicles are available in s at i
        if self.available_vehicles[s, i] >= demand_si:
            # if so, subtract the demand and go on to next vertex
            self.available_vehicles[s, i] -= demand_si
        else:
            # otherwise, compute how many vehicles are needed additionally
            needed_vehicles = demand_si - self.available_vehicles[s, i]
            # reduce available vehicles to zero as all are used
            self.available_vehicles[s, i] = 0.0
            # try to get the needed vehicles from other zones and
            # safe the number in a new local variable
            relocated_vehicles = self._relocation(needed_vehicles, s, i)
            # acquire additional vehicles if needed and add them
            # to the previous flow
            vehicles_to_add = needed_vehicles - relocated_vehicles
            if vehicles_to_add > 0.0:
                self._acquire_new_vehicles(s, i, vehicles_to_add)

    def _relocation(self, needed_vehicles, s, i):
        # variable to track how many vehicles were found
        relocated_vehicles = 0.0

        # check first the neighboring zones that are close
        for dist in range(1, min(self.max_relocation_distance, i) + 1):
            if self.all_neighbours_to.get(s).get(dist) is None:
                self.all_neighbours_to.get(s).update({dist: set()})

            # Check all neighbours that are exactly dist far away
            for neighbour in self.all_neighbours_to.get(s).get(dist):
                # What is the maximal number of vehicles available in the
                # (dist) last time intervals?
                available_at_neighbour = sys.float_info.max
                for time in range(i - dist, i):
                    available_at_neighbour = min(available_at_neighbour, self.available_vehicles[neighbour, time])

                # Compute how many vehicles are available in current time slice
                not_needed_vehicles_at_neighbour = 0

                # If neighbour was already treated, check the available vehicles there
                if neighbour < s:
                    not_needed_vehicles_at_neighbour = self.available_vehicles[neighbour, i]

                # If neighbour was not treated yet, leave enough for the outgoing demand there
                else:
                    outgoing_demand = self.demand_matrix[neighbour, :, i].sum()
                    not_needed_vehicles_at_neighbour = max(0.0, (self.available_vehicles[neighbour, i] - outgoing_demand))

                available_at_neighbour = min(available_at_neighbour, not_needed_vehicles_at_neighbour)

                # Check whether more vehicles than needed are found and take only as many as needed
                if (relocated_vehicles + available_at_neighbour) >= needed_vehicles:
                    available_at_neighbour = needed_vehicles - relocated_vehicles

                # Take vehicles that are available at the neighbor and undo routing
                for time in range(i - dist, i):
                    self.available_vehicles[neighbour, time] -= available_at_neighbour
                    self.vehicle_matrix[neighbour, neighbour, time] -= available_at_neighbour

                self.available_vehicles[neighbour, i] -= available_at_neighbour

                # Consider also current time slice: If neighbour was already
                # treated, remove routing also there as well as the available
                # vehicles in the following time slice
                if neighbour < s:
                    if ((i + 1) < self.max_n_intervals):
                        self.available_vehicles[neighbour, i + 1] -= available_at_neighbour

                    self.vehicle_matrix[neighbour, neighbour, i] -= available_at_neighbour

                # Route vehicles that are available at the neighbor to current zone
                self.vehicle_matrix[neighbour, s, (i - dist)] += available_at_neighbour

                # Add them to found vehicles
                relocated_vehicles += available_at_neighbour

                # Return early if sufficiently many vehicles are found
                if relocated_vehicles == needed_vehicles:
                    return relocated_vehicles
        return relocated_vehicles

    def _acquire_new_vehicles(self, s, i, additional_vehicles):
        # Add the acquired vehicles to the current zone already
        # in the first time interval let them stay there
        for j in range(i):
            self.vehicle_matrix[s, s, j] += additional_vehicles

    def check_feasibility(self):
        n_warnings = 0
        is_feasible = True
        warnings = ""

        # check whether flow is non negative / demand is unmet
        negative_flow = np.argwhere(self.vehicle_matrix - self.demand_matrix < 0)
        if len(negative_flow) > 0:
            is_feasible = False
            for i in negative_flow:
                warnings += "\n" + "Flow negative/unmet demand at arc " + str(i)
                n_warnings += 2

        # check whether flow is consistent
        for s in range(self.nr_zones):
            for i in range(1, self.max_n_intervals):
                incoming = 0
                outgoing = 0

                # Compute incoming and outgoing flow
                dist_values = self.distance_matrix[:, s]
                dist_values_test = -1 * dist_values + i
                dist_values_index_t = np.argwhere(dist_values_test >= 0)
                dist_values_test = dist_values_test[dist_values_index_t]

                incoming += self.vehicle_matrix[dist_values_index_t.tolist(),s, dist_values_test.tolist()].sum()
                outgoing += self.vehicle_matrix[s, :, i].sum()
                if (incoming != outgoing):
                    warnings += "\n" + "Flow not consistent in the following event (s,i) = (" + str((s+1)) + "," + str((i+1)) + "): in=" + str(incoming) + "!=" + str(outgoing) + "=out|\n"
                    n_warnings += 1
                    is_feasible = False

        return is_feasible, warnings


def solution_df_function(z, nr_intervals, i_matrix, s_matrix, dist_df):
    # function to create solution dataframe with columns from, to, t (date), veh, demand, time, time_intervals, and distance
    # inputs are
    # z: dataframe with one columns being the zone id
    # nr_intervals: the number of intervals
    # i_matrix: the input_matrix
    # s_matrix: the solution_matrix
    # dist_df: a dataframe with columns from (zone id), to (zone id), time (real trip times), time_intervals (descrete time), distance (distance between zones)
    nr_zones = len(z)
    df_int = pd.DataFrame(list(range(0, nr_intervals)))
    df_int.rename(columns={0: 't'}, inplace=True)
    int_sticker = pd.concat([df_int]*nr_zones*nr_zones, ignore_index=True)
    int_sticker['ind'] = int_sticker.index

    df_to = pd.DataFrame(data=z.zone_id)
    df_to = pd.DataFrame(np.repeat([df_to],nr_intervals))
    to_sticker = pd.concat([df_to]*nr_zones, ignore_index=True)
    to_sticker.rename(columns={0: 'to'}, inplace=True)
    to_sticker['ind'] = to_sticker.index

    df_from = pd.DataFrame(data=z.zone_id)
    from_sticker = pd.DataFrame(np.repeat([df_from],nr_intervals*nr_zones))
    # from_sticker = pd.concat([df_from], ignore_index=True)
    from_sticker.rename(columns={0: 'from'}, inplace=True)
    from_sticker['ind'] = from_sticker.index

    df_solution = pd.DataFrame(data=s_matrix.reshape(nr_zones*nr_zones*nr_intervals))
    df_solution['ind'] = df_solution.index
    df_solution.rename(columns={0: 'veh'}, inplace=True)

    df_input = pd.DataFrame(data=i_matrix.reshape(nr_zones*nr_zones*nr_intervals))
    df_input['ind'] = df_input.index
    df_input.rename(columns={0: 'demand'}, inplace=True)

    solution_df = pd.merge(pd.merge(pd.merge(pd.merge(from_sticker, to_sticker, on = "ind"), int_sticker, on = "ind"), df_solution, on = "ind"), df_input, on = "ind")
    solution_df = solution_df.merge(dist_df, on = ['from', 'to']).drop(['ind'], axis = 1)
    return solution_df


def indicators_columns(df, interval_duration):
    # function to add columns to calculate indicators for discrete and real time values, returns the solution dataframe with added colums for indicators    
    solution_df_with_indicators = df.copy()
    solution_df_with_indicators["idle_veh"] = (solution_df_with_indicators["from"] == solution_df_with_indicators["to"])*(solution_df_with_indicators["veh"]-solution_df_with_indicators["demand"])
    solution_df_with_indicators["discrete_t_veh"] = interval_duration*solution_df_with_indicators["time_intervals"]*solution_df_with_indicators["veh"]
    solution_df_with_indicators["discrete_t_demand"] = interval_duration*solution_df_with_indicators["time_intervals"]*solution_df_with_indicators["demand"]

    solution_df_with_indicators["real_t_demand"] = solution_df_with_indicators["time"]*solution_df_with_indicators["demand"] #also real time spent by vehicle with someone in it
    solution_df_with_indicators["real_t_veh_moving"] = solution_df_with_indicators["time"]*(solution_df_with_indicators["veh"]-solution_df_with_indicators["idle_veh"])
    solution_df_with_indicators["real_t_veh_moving_empty"] = solution_df_with_indicators["real_t_veh_moving"] - solution_df_with_indicators["real_t_demand"]
    solution_df_with_indicators["real_t_veh_not_moving"] = solution_df_with_indicators["discrete_t_veh"]*interval_duration - solution_df_with_indicators["real_t_veh_moving"]
    # veh not moving not ok, because we do not see all vehicles in matrix once intervals are small enough for vehicle to need two time intervals to perform one trip.

    solution_df_with_indicators["discrete_t_empty"] = interval_duration*(solution_df_with_indicators["discrete_t_veh"] - solution_df_with_indicators["discrete_t_demand"])

    # time spent on empty vehicles
    # solution_df_with_indicators["real_t_empty"] = solution_df_with_indicators["real_t_veh"] - solution_df_with_indicators["real_t_demand"] + solution_df_with_indicators["idle_veh"]*interval_duration
    solution_df_with_indicators["passenger_km"] = solution_df_with_indicators["distance"]*solution_df_with_indicators["demand"]
    solution_df_with_indicators["vehicle_km"] = solution_df_with_indicators["distance"]*(solution_df_with_indicators["veh"]-solution_df_with_indicators["idle_veh"])
    solution_df_with_indicators["khlp"] = solution_df_with_indicators["vehicle_km"] - solution_df_with_indicators["passenger_km"]
    solution_df_with_indicators["empty_veh_km"] = solution_df_with_indicators["vehicle_km"] - solution_df_with_indicators["passenger_km"]

    return solution_df_with_indicators


def indicators(solution_df_with_indicators_columns):
    # Calculate indicators (tps_roulage_plein, tps_roulage_tot, tps_roulage_vide, tps_veh_immobile, kcc, ktot, pourcentage_kcc)
    tps_roulage_plein = solution_df_with_indicators_columns['real_t_demand'].sum()
    tps_roulage_tot = solution_df_with_indicators_columns['real_t_veh_moving'].sum()
    tps_roulage_vide = solution_df_with_indicators_columns['real_t_veh_moving_empty'].sum()
    tps_veh_immobile = solution_df_with_indicators_columns["discrete_t_veh"].sum() - tps_roulage_tot # or solution_df_with_indicators["real_t_veh_not_moving"]/sum()
    # careful, no direct relationship with number_of_vehicle(t) * timeinterval, as number_of_vehicle(t) is not constant with t
    kcc = solution_df_with_indicators_columns['passenger_km'].sum()
    ktot = solution_df_with_indicators_columns['vehicle_km'].sum()
    pourcentage_kcc = kcc/ktot

    return (
        tps_roulage_plein,
        tps_roulage_tot,
        tps_roulage_vide,
        tps_veh_immobile,
        kcc,
        ktot,
        pourcentage_kcc
    )