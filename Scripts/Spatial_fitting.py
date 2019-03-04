import numpy as np
import pandas as pd
import re
from functools import partial


# Take temporally fitted data and fit it spatially

def parse_this(s: str):
    try:
        x = float(s)
        return x
    except:
        try:
            x = tuple([float(re.sub('[)( ]', '', e)) for e in s.split(',')])
            return x
        except:
            return s


def convert_to_tuple(element):
    try:
        element = element.str.lstrip(' ()').str.rstrip(' ()')
        x = [float(e) for e in element.str.split(',')]
        return x
    except:
        try:
            element = element.lstrip(' ()').rstrip(' ()')
            x = [float(e) for e in element.split(',')]
            return x
        except:
            return element


def data_from_csv(filename1, filename2, sep=';', video=False):
    df1 = pd.read_csv(filename1, sep=sep, converters={i: str for i in range(
        500)})
    df2 = pd.read_csv(filename2, sep=sep, converters={i: str for i in range(
        500)})
    df1 = df1.dropna()
    df2 = df2.dropna()
    if video == True:
        df1 = df1.iloc[:, 1:]
        df2 = df2.iloc[:, 1:]
    df1.apply(convert_to_tuple)
    df2.apply(convert_to_tuple)
    return df1, df2


def projection(df1_3d, df2_3d, vertical, label='Hip'):
    hip1 = np.asarray(convert_to_tuple(df1_3d[label].values[0]))
    hip2 = np.asarray(convert_to_tuple(df2_3d[label].values[0]))
    people_vector = hip1 - hip2
    cross = np.cross(people_vector, vertical)
    print(hip1, hip2, people_vector, cross)
    B = np.column_stack((people_vector, vertical, cross))
    P = np.array([[1, 0, 0], [0, 1, 0]])
    print('B: ', B)
    print('P: ', P)

    # Define corresponding 2D dataframes after projection
    projected_df1, projected_df2 = df1_3d.copy(True), df2_3d.copy(True)
    projected_df1 = projected_df1.apply(convert_to_tuple)
    projected_df2 = projected_df2.apply(convert_to_tuple)

    def projection(x, P, B):
        try:
            res = np.dot(np.dot(P, np.linalg.inv(B)), np.asarray(x,
                                                                 dtype=np.float64))
            return res
        except:
            return x

    for col in projected_df1.columns:
        if col == 'Type' or col == 'Time':
            continue
        projected_df1[col] = projected_df1[col].apply(projection, args=(P, B))
    for col in projected_df2.columns:
        if col == 'Type' or col == 'Time':
            continue
        projected_df1[col] = projected_df2[col].apply(projection, args=(P, B))
    return projected_df1, projected_df2


def rescale(df):
    return rescaled_df


def fit(df1, df2):
    return transformed_df1, df2


if __name__ == '__main__':
    video1, video2 = data_from_csv('../Data/video_coordinates_tuples_1.csv',
                                   '../Data/video_coordinates_tuples_1.csv',
                                   video=True, sep=';')
    print(video1)
    mocap1, mocap2 = data_from_csv('../Data/Mocap_tuples1.csv',
                                   '../Data/Mocap_tuples2.csv', sep=';')
    print(mocap1)

    vertical = np.array([0, 1, 0])  # TODO check which vector is vertical
    mocap1_2d, mocap2_2d, = projection(mocap1, mocap2, vertical, label='Hip')
