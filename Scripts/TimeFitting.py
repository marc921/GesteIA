import math
import numpy as np
import pandas as pd
from tqdm import tqdm


def normalize_array(x):
    x = np.array(x)
    max_elt = x.max()
    x = x / max_elt
    return x


def interpolate(df, t):
    if len(df[df['Time'] == t]) == 0:
        point_inf = df[df['Time'] < t]
        point_inf = point_inf[point_inf['Time'] == point_inf['Time'].max()]

        point_sup = df[df['Time'] > t]
        point_sup = point_sup[point_sup['Time'] == point_sup['Time'].min()]

        # checking borders
        if len(point_inf) == 0 and len(point_sup) != 0:
            return point_sup['Data'].values[0]
        elif len(point_inf) != 0 and len(point_sup) == 0:
            return point_inf['Data'].values[0]
        elif len(point_inf) == 0 and len(point_sup) == 0:
            return 2

        t_inf = point_inf['Time'].values[0]
        t_sup = point_sup['Time'].values[0]

        v_inf = point_inf['Data'].values[0]
        v_sup = point_sup['Data'].values[0]

        v = v_inf + ((t - t_inf) / (t_sup - t_inf)) * v_sup
        return v
    else:
        return df[df['Time'] == t]['Data'].values[0]


def similarity_cost(df1, df2):
    """
        Input : DataFrame with columns ['Time', 'Data']
    """
    df1.reset_index(inplace=True)
    df2.reset_index(inplace=True)

    cost = 0
    for i in range(len(df1)):
        v1 = df1.loc[i, 'Data']
        t = df1.loc[i, 'Time']

        v2 = interpolate(df2, t)

        cost += (v1 - v2)**4

    return cost


def compare_on_window(df1, df2, time_window=0.5, time_step=-1):
    """
        Input : DataFrame with columns ['Time', 'Data']
    """
    if time_step == -1:
        time_step = time_window / 5

    t_min = max(df1['Time'].min(), df2['Time'].min())
    t_max = min(df1['Time'].max(), df2['Time'].max())

    time_step_cpt = 0
    total_cost = 0
    for t_idx in tqdm(np.arange(t_min, t_max - time_window, time_step), desc=' ', leave=False):
        time_slice_1 = df1[df1['Time'] >= t_idx]
        time_slice_1 = time_slice_1[time_slice_1['Time'] < (t_idx + time_window)]
        time_slice_2 = df2[df2['Time'] >= t_idx]
        time_slice_2 = time_slice_2[time_slice_2['Time'] < (t_idx + time_window)]

        total_cost += similarity_cost(time_slice_1, time_slice_2)
        time_step_cpt += 1

    return total_cost / time_step_cpt
