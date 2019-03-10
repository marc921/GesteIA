import numpy as np
import pandas as pd
import os, sys, re
import math
from functools import partial
from scipy.optimize import minimize


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


def convert_el_to_tuple(element):
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


def convert_to_tuple(df):
    for col in df.columns:
        if col == 'Type':
            continue
        elif col in ['time', 'Time']:
            df[col] = pd.to_numeric(df[col])
        else:
            df[col] = df[col].apply(convert_el_to_tuple)
    return df


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
    df1 = convert_to_tuple(df1)
    df2 = convert_to_tuple(df2)
    return df1, df2


def preprocess_mocap(df):
    processed_df = df.copy(True)
    processed_df = processed_df[processed_df['Type'] == 'position'].dropna()
    processed_df = convert_to_tuple(processed_df)
    return processed_df


def preprocess_video(df):
    processed_df = df.copy(True)
    processed_df = processed_df.dropna()
    processed_df = processed_df.rename(columns={'MidHip': 'Hip'})
    processed_df = convert_to_tuple(processed_df)
    return processed_df


def project3d_to_plane(df1_3d, df2_3d, vertical, label='Hip'):
    hip1 = np.asarray(convert_el_to_tuple(df1_3d[label].values[0]))
    hip2 = np.asarray(convert_el_to_tuple(df2_3d[label].values[0]))
    people_vector = hip1 - hip2
    cross = np.cross(people_vector, vertical)
    print('Projecting 3d pointcloud to plane', hip1, hip2, people_vector, cross)
    B = np.column_stack((people_vector, vertical, cross))
    P = np.array([[1, 0, 0], [0, 1, 0]])
    print('B: ', B)
    print('P: ', P)

    # Define corresponding 2D dataframes after projection
    projected_positions1 = preprocess_mocap(df1_3d)
    projected_positions2 = preprocess_mocap(df2_3d)

    def projection(x, P, B):
        try:
            res = np.dot(np.dot(P, np.linalg.inv(B)), np.asarray(x,
                                                                 dtype=np.float64))
            return res
        except:
            return x

    for col in projected_positions1.columns:
        if col in ['Type', 'Time', 'time']:
            continue
        projected_positions1[col] = projected_positions1[col].apply(projection,
                                                                    args=(P, B))
    for col in projected_positions2.columns:
        if col in ['Type', 'Time', 'time']:
            continue
        projected_positions2[col] = projected_positions2[col].apply(projection,
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
        if col in ['Type', 'Time', 'time']:
            continue
        df[col] = df[col].apply(rotate_point, args=(theta, vertical))
    return df


def rescale(df, scaling_factor):
    for col in df.columns:
        if col in ['Type', 'Time', 'time']:
            continue
        df[col] = df[col].apply(lambda x: scaling_factor * x)
    return rescaled_df


def L2_distance_df(df1, df2, video=False):
    if video == True:
        df1 = preprocess_video(df1)
        df2 = preprocess_video(df2)
    else:
        df1 = preprocess_mocap(df1)
        df2 = preprocess_mocap(df2)
    dist = 0
    for col in df1.columns:
        if col in ['Type', 'Time', 'time']:
            continue
        elif col in df2.columns:
            for i in range(len(df1[col].values)):
                dist += np.linalg.norm(
                    np.asarray(df1[col].values[i]) - np.asarray(
                        df2[col].values[i]))
        else:
            continue
    return np.sqrt(dist)


def fit_translate(mocap, video, ref_time=120.0, label='Hip', sampleframe=False):
    if sampleframe == False:
        video_min = video.loc[video['time'] <= ref_time]
        video_ref = video_min.loc[video_min['time'].idxmax()]
        mocap_min = mocap.loc[mocap['Time'] <= ref_time]
        mocap_ref = mocap_min.loc[mocap_min['Time'].idxmax()]
    else:
        video_ref = video.iloc[0]
        mocap_ref = mocap.iloc[0]
    T = np.asarray(
        convert_el_to_tuple(video_ref[label])) - np.asarray(
        convert_el_to_tuple(mocap_ref[label]))
    fitted_mocap = mocap.copy(True)

    # TODO might not be precise enough?

    def translate_point(point, T):
        try:
            x = point + T
            return x
        except:
            return point

    for col in fitted_mocap.columns:
        if col in ['Type', 'Time', 'time']:
            continue
        fitted_mocap[col] = fitted_mocap[col].apply(translate_point, args=(T,))
    return fitted_mocap, T


def optimize_projection(mocap1, mocap2, video1, video2, vertical,
                        label='Hip', verbose=False, sampleframe=False):
    # Only optimize pointcloud adjustment over a few samples
    if sampleframe == False:
        mocap1 = preprocess_mocap(mocap1).iloc[0:500:10]
        video1 = preprocess_video(video1).iloc[0:500:10]
        mocap2 = preprocess_mocap(mocap2).iloc[0:500:10]
        video2 = preprocess_video(video2).iloc[0:500:10]

    def score_projection(theta, vertical, mocap1, mocap2, video1,
                         video2, label):
        if verbose:
            print('Before rotation: ', mocap1, mocap2)
        rotated_mocap1 = rotate_data(mocap1, theta[0], vertical)
        rotated_mocap2 = rotate_data(mocap2, theta[1], vertical)
        projected_mocap1, projected_mocap2 = project3d_to_plane(
            rotated_mocap1,
            rotated_mocap2,
            vertical,
            label=label)
        if verbose:
            print('After projection: ', projected_mocap1, projected_mocap2)
        translated_mocap1, T1 = fit_translate(projected_mocap1, video1,
                                              label=label)
        translated_mocap2, T2 = fit_translate(projected_mocap2, video2,
                                              label=label)
        if verbose:
            print('After macro fitting: ', translated_mocap1,
                  translated_mocap2)
        # TODO check translation after projection ok?!
        score = L2_distance_df(translated_mocap1, video1, video=True) + \
                L2_distance_df(translated_mocap2, video2, video=True)
        return score

    optim = minimize(score_projection, np.array([0.0, 0.0]),
                     args=(vertical, mocap1, mocap2, video1, video2, label))
    return optim


if __name__ == '__main__':
    sampleframe = bool(input('Only testing with a few frames (True) or with '
                             'the whole database (False)? '))

    # Retrieve data from video and Mocap
    video1, video2 = data_from_csv('Data/video_coordinates_tuples_1.csv',
                                   'Data/video_coordinates_tuples_2.csv',
                                   video=True, sep=';')
    if sampleframe:
        video1 = video1.iloc[[675, 1243]]
        video2 = video2.iloc[[675, 1243]]
    video1 = preprocess_video(video1)
    video2 = preprocess_video(video2)
    print('Video data 1: ', video1)
    print('Video data 2: ', video2)
    mocap1, mocap2 = data_from_csv('Data/Mocap_tuples1.csv',
                                   'Data/Mocap_tuples2.csv', sep=';')
    # Mocap coord syst from camera: x going left, y vertical, z going outward
    if sampleframe:
        mocap1 = mocap1.iloc[[1492, 2824]]
        mocap2 = mocap2.iloc[[1492, 2824]]
    mocap1 = preprocess_mocap(mocap1)
    mocap2 = preprocess_mocap(mocap2)
    print('Original mocap data 1: ', mocap1)
    print('Original mocap data 2: ', mocap2)

    # Unit tests
    vertical = np.array([0, 1, 0])
    print(rotate_point(np.array([1, 0, 0]), 1.57, np.array([0, 0, 1])))
    print(np.linalg.norm(np.asarray(video1['RShoulder'].values[0]) -
                         np.asarray(video2['RShoulder'].values[0])))
    print(L2_distance_df(video2, video1, video=True))
    # TODO look at source data!

    # Find optimal initial transformation of pointclouds before projection
    optim = optimize_projection(mocap1,
                                mocap2,
                                video1,
                                video2,
                                vertical,
                                label='Hip',
                                verbose=False,
                                sampleframe=sampleframe)
    print('Optimization result: ', optim)
    rotated_mocap1 = rotate_data(mocap1, optim.x[0], vertical)
    rotated_mocap2 = rotate_data(mocap2, optim.x[1], vertical)

    # Project macrofitted pointclouds onto 2D person plane
    projected_mocap1, projected_mocap2 = project3d_to_plane(rotated_mocap1,
                                                            rotated_mocap2,
                                                            vertical,
                                                            label='Hip')
    fitted_mocap1, T1 = fit_translate(projected_mocap1, video1, label='Hip')
    fitted_mocap2, T2 = fit_translate(projected_mocap2, video2, label='Hip')

    # Record results
    os.makedirs('Data/Results', exist_ok=False)
    fitted_mocap1.to_csv('Data/Results/Projected_mocap1.csv', sep=';')
    fitted_mocap2.to_csv('Data/Results/Projected_mocap2.csv', sep=';')
    print('Projected mocap data 1: ', fitted_mocap1)
    print('Projected mocap data 2: ', fitted_mocap2)
    with open('Data/Results/Optim_results.txt', 'w') as text_file:
        print(optim, file=text_file)
        print('', file=text_file)
        print('Translation of mocap 1', T1, file=text_file)
        print('Translation of mocap 2', T2, file=text_file)
