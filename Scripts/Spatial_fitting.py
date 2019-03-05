import numpy as np
import pandas as pd
import re
import scipy as sp
import math
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


def project3d_to_plane(df1_3d, df2_3d, vertical, label='Hip'):
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
    projected_positions1, projected_positions2 = df1_3d.copy(True), \
                                                 df2_3d.copy(True)
    projected_positions1 = projected_positions1[projected_positions1['Type']
                                                == 'position'].dropna()
    projected_positions2 = projected_positions2[projected_positions2['Type']
                                                == 'position'].dropna()
    projected_positions1 = projected_positions1.apply(convert_to_tuple)
    projected_positions2 = projected_positions2.apply(convert_to_tuple)

    def projection(x, P, B):
        try:
            res = np.dot(np.dot(P, np.linalg.inv(B)), np.asarray(x,
                                                                 dtype=np.float64))
            return res
        except:
            return x

    for col in projected_positions1.columns:
        if col == 'Type' or col == 'Time':
            continue
        projected_positions1[col] = projected_positions1[col].apply(projection,
                                                                    args=(P, B))
    for col in projected_positions2.columns:
        if col == 'Type' or col == 'Time':
            continue
        projected_positions1[col] = projected_positions2[col].apply(projection,
                                                                    args=(P, B))
    projected_positions1 = projected_positions1.dropna()
    projected_positions2 = projected_positions2.dropna()
    return projected_positions1, projected_positions2


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    https://stackoverflow.com/questions/6802577/rotation-of-3d-vector
    """
    axis = np.asarray(axis)
    axis = axis / np.linalg.norm(axis)
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def rotate_point(point, theta, vertical):
    try:
        rotated_point = np.dot(rotation_matrix(vertical, theta), point)
        return rotated_point
    except:
        return point


def rotate_data(df, theta, vertical):
    for col in df.columns:
        if col == 'Type' or col == 'Time':
            continue
        df[col] = df[col].apply(rotate_point, args=(theta, vertical))


def rescale(df, scaling_factor):
    for col in df.columns:
        if col == 'Type' or col == 'Time':
            continue
        df[col] = df[col].apply(lambda x: scaling_factor * x)
    return rescaled_df


def fit_translate(mocap, video, label='Hip'):
    T = np.asarray(convert_to_tuple(video[label].values[0])) - np.asarray(
        convert_to_tuple(mocap[label].values[0]))
    print(T, type(T))

    def translate_point(point, T):
        try:
            return point + T
        except:
            return point

    for col in mocap.columns:
        if col == 'Type' or col == 'Time':
            continue
        mocap[col] = mocap[col].apply(translate_point, args=T)
    return fitted_mocap, video


def optimize_projection(mocap, video, vertical, label='Hip'):
    # TODO

    def score_projection(theta, vertical, mocap, video):
        rotate_data(mocap, theta, vertical)

    optim = sp.optimize.minimize(score_projection,
                                 args=(vertical, mocap, video))
    return projected_mocap


if __name__ == '__main__':
    # Unit tests
    print(rotate_point(np.array([1, 0, 0]), 1.57, np.array([0, 0, 1])))

    # Retrieve data
    video1, video2 = data_from_csv('../Data/video_coordinates_tuples_1.csv',
                                   '../Data/video_coordinates_tuples_1.csv',
                                   video=True, sep=';')
    print('Video data 1: ', video1)
    mocap1, mocap2 = data_from_csv('../Data/Mocap_tuples1.csv',
                                   '../Data/Mocap_tuples2.csv', sep=';')
    # Mocap coord syst: x outward from person, y vertical, z from left to
    # right shoulder
    print('Original mocap data 1: ', mocap1)

    # Test: project Mocap data directly onto 2D plane
    vertical = np.array([0, 1, 0])
    mocap_positions1_2d, mocap_positions2_2d, = project3d_to_plane(mocap1,
                                                                   mocap2,
                                                                   vertical,
                                                                   label='Hip')
    print('Projected mocap data 1: ', mocap_positions1_2d)

    # Find optimal initial transformation of pointclouds before projection
