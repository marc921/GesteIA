import numpy as np
import pandas as pd

# Take temporally fitted data and fit it spatially

def data_from_csv(filename1, filename2, sep=',', video=False):
    df1 = pd.read_csv(filename1, sep=sep)
    df2 = pd.read_csv(filename2, sep=sep)
    df1 = df1.dropna()
    df2 = df2.dropna()
    if video == True:
        df1 = df1.ix[:, 1:]
        df2 = df2.ix[:, 1:]
    return df1, df2

def projection(df1_3d, df2_3d, vertical, label='Mid_hip'):
    hip1 = np.array(df1_3d[label].iloc[0].values)
    hip2 = np.array(df2_3d[label].iloc[0].values)
    people_vector = hip1 - hip2
    cross = np.cross(people_vector, vertical)
    print(hip1, hip2, people_vector, cross)
    B = np.concatenate((people_vector, vertical, cross), axis=1)
    P = np.array([1, 0, 0], [0, 1, 0])

    # Define corresponding 2D dataframes after projection
    projected_df1, projected_df2 = df1_3d, df2_3d
    # z1_cols = [c for c in projected_df1.columns if c.lower()[-2:] != '_z']
    # projected_df1 = projected_df1[z1_cols]
    # z2_cols = [c for c in projected_df2.columns if c.lower()[-2:] != '_z']
    # projected_df2 = projected_df2[z2_cols]
    for col in df1_3d.columns:
        point = np.array(df1_3d[col])
        projected_df1[col] = np.dot(np.dot(P, np.inv(B)), point)
    for col in df2_3d.columns:
        point = np.array(df2_3d[col])
        projected_df2[col] = np.dot(np.dot(P, np.inv(B)), point)

    return projected_df1, projected_df2

def rescale(df):
    return rescaled_df

def fit(df1, df2):

    return transformed_df1, df2


if __name__ == '__main__':
    video1, video2 = data_from_csv('../Data/video_coordinates_1.csv',
                                   '../Data/video_coordinates_2.csv',
                                   video=True)
    print(video1)
    mocap1, mocap2 = data_from_csv('../Data/Mocap_1.csv',
                                   '../Data/Mocap_2.csv', sep=';')
    print(mocap1)