import random
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

default_parameters = {
    'max_waiting_duration_min': 15,
    'vehicle_capacity': 10,
    'demand_duration': 60,
    'as_integers': True,
    'vehicle_demand_to_integer': True,
    'aggregate_method': 'logic',
}


class VSA_Preparation():

    def __init__(self, zones, volumes, distances, **params_kwargs):
        self.zones = zones
        self.volumes = volumes
        self.distances = distances
        self.parameters = default_parameters.copy()
        # print(params_kwargs)
        for k, v in params_kwargs.items():
            self.parameters[k] = v

        # preprocessing
        self._clean()
        self.trip_step = round(self.distances['time'].quantile(0.1))
        self.n_intervals = int(self.parameters['demand_duration'] / self.trip_step)

        if self.parameters['max_waiting_duration_min'] is not None:
            self.demand_step = int(
                np.floor(self.parameters['max_waiting_duration_min'] / self.trip_step)
            )
            self.demand_step = max(self.demand_step, 1)

    def set_parameters(self, **params_kwargs):
        for k, v in params_kwargs.items():
            self.parameters[k] = v
        self.demand_step = int(
            np.floor(self.parameters['max_waiting_duration_min'] / self.trip_step)
        )
        self.n_intervals = int(self.parameters['demand_duration'] / self.trip_step)

    def _clean(self):
        self.volumes.dropna(inplace=True)
        # self.distances = self.distances.loc[self.volumes.index]

    def discretize_aggregate(self):
        # discretize distances
        self.distances['time_interval'] = (self.distances['time'] / self.trip_step).apply(np.ceil).clip(1) # 'time' and not 'distance'

        # split volumes
        random.seed(1)  # to ensure reproducibility in integer demand split
        self.volumes_time = self.volumes['volume'].apply(
            lambda x: split_volume(x, self.n_intervals, self.parameters['as_integers'])
        )
        # convert to vehicle volumes
        self.vehicle_volumes_time = self.volumes_time / self.parameters['vehicle_capacity']

        # aggregate volumes
        if self.parameters['aggregate_method'] is not None:
            self.vehicle_volumes_time = self.vehicle_volumes_time.apply(
                aggregate_demand, distance_max=self.demand_step, method=self.parameters['aggregate_method']
            )

        # TODO: aggregate zones here:
        # - cluster zones based on time-distance (< max waiting time)
        # - aggregate volumes for each OD cluster
        # - questions:
        #       Time-distance penalty for aggregated zones?
        #       How to compute time-distance between clusters?

        # convert to full vehicles
        if self.parameters['vehicle_demand_to_integer']:
            self.vehicle_volumes_time = self.vehicle_volumes_time.apply(
                lambda x: np.ceil(x)
            )

    def compute_vsa_inputs(self):
        n_zones = len(self.zones)
        # Format inputs
        distance_matrix_input = self.distances['time_interval'].map(int).values.reshape(n_zones, n_zones)

        odv_time_input = self.vehicle_volumes_time.reindex(
            pd.MultiIndex.from_product(
                [self.zones['zone_id'], self.zones['zone_id']], names=['from', 'to']),
            fill_value=np.zeros(self.n_intervals)
        )

        demand_matrix_input = np.array([np.array(x) for x in odv_time_input.values])
        demand_matrix_input = demand_matrix_input.reshape(n_zones, n_zones, self.n_intervals)
        demand_matrix_input[demand_matrix_input < 0.0] = 0.0

        return distance_matrix_input, demand_matrix_input


def _split_volume_equally(x, n):
    return np.full(n, x/n)


def _split_volume_integers(x, n):
    """
    Distribute demand x over  n intervals.
    Returns an array with the demand for each interval
    """
    # create array with the same value for each interval (x//n)
    s = np.full(n, x//n)
    # randomly add the remaining demand to the intervals
    choices = random.sample(range(n), x % n)
    for i in choices:
        s[i] += 1

    return s


def split_volume(x, n, as_integers=True):
    if as_integers:
        return _split_volume_integers(int(np.ceil(x)), n)
    else:
        return _split_volume_equally(x, n)


def _aggregate_demand_logic(s, distance_max):
    agg_s = np.zeros(len(s))
    i = 0
    agg = 0
    while i < len(s):
        count = 0
        latest = i
        for j in range(i, i+distance_max):
            if j < len(s):
                count += 1
                agg += s[j]
                if agg == 0 or agg >= 1:
                    agg_s[j] = agg//1
                    agg = agg % 1
                    i += 1
                    break
                elif count == distance_max:
                    if s[j] > 0:
                        latest = j
                    agg_s[latest] = agg
                    agg = 0
                    i += 1
                    break
                else:
                    if s[j] > 0:
                        latest = j
                    i += 1
            else:
                agg_s[j-1] = agg
                i += 1
                break
    return agg_s


def _aggregate_demand_cluster(s, distance_max):
    # split demand into full vehicles and remaining demand
    full_vehicles = s//1
    s = s % 1
    # group remaining demand
    # create clusters within s using agglomerative clustering and max distance 3
    if np.count_nonzero(s) < 2:
        return s
    clustering = AgglomerativeClustering(
        n_clusters=None, distance_threshold=distance_max).fit_predict(
            np.argwhere(s.reshape(-1, 1) > 0)
    )
    d = {i: np.argwhere(clustering == i).flatten() for i in np.unique(clustering)}
    add = {v.max(): s[v].sum() for k, v in d.items()}

    for k, v in add.items():
        full_vehicles[k] += v

    return full_vehicles


def aggregate_demand(s, distance_max, method='logic'):
    if method == 'logic':
        return _aggregate_demand_logic(s, distance_max)
    elif method == 'cluster':
        return _aggregate_demand_cluster(s, distance_max)
    else:
        raise ValueError('Method not supported. Choose between logic and cluster.')
