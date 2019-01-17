# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 14:39:18 2018

@author: sacha gobeyn (sachagobeyn@gmail.com, sacha@fluves.com)
"""
import pandas as pd
import numpy as np
import os
import datetime
from copy import deepcopy


#2014-01-01 00:00:00
def ComputeRainfallErosivity(fname, full_write=True, delimiter=",", dformat="%Y-%m-%d %H:%M:%S",test_flag=False):
    """ Script to compute rainfall erosivity factor R
    Calculations based on formula's reported in
    Panagos, P., Ballabio, C., Borrelli, P., Meusburger, K., Klik, A., Rousseva, S., … Alewell, C., 2015, Rainfall erosivity in Europe. Science of the Total Environment, 511, 801–814.
    Verstraeten, G., Poesen, J., Demarée, G., & Salles, C., 2006, Long-term (105 years) variability in rain erosivity as derived from 10-min rainfall depth data for Ukkel (Brussels, Belgium): Implications for assessing soil erosion rates. Journal of Geophysical Research Atmospheres, 111(22), 1–11.

    Parameters
    ----------
        'fname' (str): name of .csv input file
            format (',' or ';', .. delimited):

            timestamp;value
            20040518161000;0
            20040518162000;0
            20040518163000;0
            ...

            with
                timestamp: YYYYMMDDHHmmSS
                value: float
        'full_write' (boolean): True (False) if all info should (not) be writen to .csv
        'delimiter' (string): delimiter of input data
        'dformat' (string): format of timestamp

    Returns
    -------
        'df' (pandas dataframe): computed R factors per year
    """

    "create directory"
    create_dir("Results", [""])

    "load and format df"
    df, dt = load_and_format_input_file(fname, delimiter, dformat, test_flag = test_flag)

    "identify erosive storms"
    df = identify_erosive_storms(df, dt)

    "calculate I30"
    df = calculate_I30(df, dt)

    "calculate EI30"
    fname  = os.path.split(fname)[-1]
    fname = fname[0:fname.index(".")]
    df = calculate_R(df, dt, full_write=full_write, fname=os.path.join("Results", fname))

    return df


def identify_erosive_storms(df, dt):
    """ Identify erosive storms under conditions
    (1) Split in storms is defined by a 6-hour precipitation total lower than 1.27 mm
    (2) A storm is a considered if
        (a) total rainfall per storm is higher than 12.7 mm
        OR
        (b) maximum intensity is higher than 6.35/15 min

    Parameters
    ----------
        'df' (pandas dataframe) :
            'N': precipitation
            'Ni': intensity
            'timestamp': YYYYMMDDHHmmSS, pandas datatime series
        'dt' (float): delta t of df (in seconds)

    Returns
    -------
        'df' (pandas dataframe): N and Ni of identified storm
    """

    #    df["stormid"] = 0
    #
    #    df.loc[df.index[1]:,"stormid"] = [1 if (df.loc[df.index[i],"timestamp"]-df.loc[df.index[i-1],"timestamp"])>=datetime.timedelta(hours=6) else 0 for i in range(1,len(df),1)]
    #
    #    df["stormid"] = df["stormid"].cumsum()

    "code based on Verstraeten, G., Poesen, J., Demarée, G., & Salles, C., 2006, Long-term (105 years) variability in " \
    "rain erosivity as derived from 10-min rainfall depth data for Ukkel (Brussels, Belgium): Implications for assessing" \
    " soil erosion rates. Journal of Geophysical Research Atmospheres, 111(22), 1–11."

    #    "calculate sum over past 6 hours for every record"
    df["6-hourN"] = 0.
    period = 6 * 3600  # 6 hours (in seconds)
    records = int(period / dt)
    df.loc[df.index[records]:, "6-hourN"] = [np.nansum(df["N"].values[df.index[i - records]:df.index[i]]) for i in
                                             range(records, len(df), 1)]
    #
    #    "if sample has a cumulatie 6-hourN below a threshold (here 0. mm as in Verstraeten et al.)
    threshold = 0.
    df["stormid"] = 0
    df.loc[df.index[1]:, "stormid"] = [
        1 if (df.loc[df.index[i - 1], "6-hourN"] > threshold) & (df.loc[df.index[i], "6-hourN"] <= threshold) else 0 for
        i in np.arange(1, len(df), 1)]
    df["stormid"] = df["stormid"].cumsum()

    "get total cumulative rainfall, maximum intensity per storm"
    df["totNStorm"] = df["N"].values
    df = df[["timestamp", "N", "Ni", "6-hourN", "stormid"]].merge(
        df.groupby('stormid').aggregate({"totNStorm": np.nansum}).reset_index(), on=["stormid"], how="left")

    "condition: total sum of event should be bigger then 1.27"  # , or maximum intensity above 6.35/15 min (25.4 mm/h)"
    #    Nithreshold = 6.35/15*4
    totNthreshold = 1.27
    #  df["cond"] = #(df["maxNiStorm"]>Nithreshold) |

    cond = (df["totNStorm"] > totNthreshold)
    df = df[cond]

    return df


def calculate_I30(df, dt):
    """ Calculate I30

    Parameters
    ----------
        'df' (pandas dataframe):
            'N': precipitation
            'Ni': intensity
            'stormid': identifier of storm
            'timestamp': YYYYMMDDHHmmSS, pandas datatime series
        'dt' (float): delta t of df (in seconds)

    Returns
    -------
        'df' (pandas dataframe): N, Ni and I30 of identified storm
    """

    "calculate I30 (based on interval of dt min, for VMM df: 10 min)"
    interval = int(30 * 60 / dt) - 1  # in seconds
    df.loc[df.index[interval:len(df)], "I30"] = [np.sum(df.loc[j - interval:j, "N"]) * 2 for j in
                                                 df.index[interval:len(df)]]
    df = df[[i for i in df.columns if i != "I30"]].merge(df.groupby("stormid").aggregate({"I30": np.max}).reset_index(),
                                                         on="stormid", how="left")

    return df


def calculate_R(df, dt, full_write=True, fname="test"):
    """ Calculate R

    Parameters
    ----------
        'df' (pandas dataframe):
            'N': precipitation (mm)
            'Ni': intensity (mm/h)
            'I30': maximum 30-minute intensity per storm (mm/h)
            'stormid': identifier of storm
            'timestamp': YYYYMMDDHHmmSS, pandas datatime series
        'dt' (float): delta t of df (in seconds)
        'full_write' (boolean): True (False) if all info should (not) be writen to .csv

    Returns
    -------
        'df' (pandas dataframe): N, Ni and I30, EI30 of identified storms
    """

    "volume rainfall during time period r (mm)"
    df["vr"] = df["N"].values
    "unit rainfalll energy (MJ ha-1 mm-1)"
    df["er"] = 11.12 * (df["Ni"].values ** 0.31)

    "calculate EI30 per storm (what with storm which go over year?"
    df["year"] = df["timestamp"].dt.year
    df["ervr"] = df["er"] * df["vr"]
    df = df.groupby(["year", "stormid"]).aggregate({"ervr": np.nansum, "I30": np.nanmax,"timestamp":np.max}).reset_index()

    df["EI30"] = df["ervr"] * df["I30"]

    "write EI30 per storm"
    if full_write == True:
        df.to_csv(os.path.join(fname + "-EI30.csv"))

    "sum up EI30 over year"
    df = df.groupby("year").aggregate({"EI30": np.nansum}).reset_index()

    "J/m2 to MJ/ha (see Verstraeten, G., Poesen, J., Demarée, G., & Salles, C. (2006). Long-term (105 years) variability " \
    "in rain erosivity as derived from 10-min rainfall depth data for Ukkel (Brussels, Belgium): Implications for assessing " \
    "soil erosion rates. Journal of Geophysical Research Atmospheres, 111(22), 1–11.)"
    df["EI30"] = df["EI30"] / 100

    if full_write == True:
        df.to_csv(os.path.join(fname + "-R.csv"))
        print("Erosivity R for timeseries: %.2f" % (np.nanmean(df["EI30"])))

    return df


def load_and_format_input_file(fname, delimiter, dformat,test_flag=False):
    """ Load and format input file

    Parameters
    ----------
        'fname' (str): name of .csv input file
            format (';' delimited):

            timestamp;value
            20040518161000;0
            20040518162000;0
            20040518163000;0
            ...

            with
                timestamp: YYYYMMDDHHmmSS
                value: float
        'delimiter' (string): delimiter of input data
        'dformat' (string): format of timestamp

    Returns
    -------
        'df' (pandas dataframe) :
            'N': precipitation (mm)
            'Ni': intensity (mm/h)
            'timestamp': YYYYMMDDHHmmSS, pandas datatime series
    """
    df = load_df(fname, delimiter, dformat)

    #(SG) only work on part of the data
    if test_flag == True:
        if len(df) < test_flag:
            ind = len(df)
        else:
            ind = 1000
    else:
        ind = len(df)

    df = df.iloc[0:ind]
    df, dt = format_df(df)

    return df, dt


def load_df(fname, delimiter, dformat):
    """ Load dataframe

    Parameters
    ----------
        'fname' (str): name of .csv input file
            format (',' delimited):

            timestamp;value
            20040518161000;0
            20040518162000;0
            20040518163000;0
            ...

            with
                timestamp: YYYYMMDDHHmmSS
                value: float

        'delimiter' (string): delimiter of input data
        'dformat' (string): format of timestamp

    Returns
    -------
        'df' (pandas dataframe) :
            'value': value of precipication (mm)
            'timestamp': YYYY:MM:DD: HH:mm:SS, pandas datatime series
    """
    df = pd.read_csv(fname, delimiter=delimiter)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format=dformat)
    df.index = np.arange(0, len(df))

    return df


def format_df(df):
    """Format dataframe

    Parameters
    ----------
        'df' (pandas dataframe) :
            'value': value of intensity
            'timestamp': YYYYMMDDHHmmSS, pandas datatime series

    Returns
    -------
        'df' (pandas dataframe) :
            'N': precipitation (mm)
            'Ni': intensity (mm/h)
            'timestamp': YYYYMMDDHHmmSS, pandas datatime series
        'dt' (float): delta t of df (in seconds)
    """
    "get deltat"
    dt = df["timestamp"].diff().values[1].astype('timedelta64[s]').astype(int)
    "precipitation volume per time step (mm)"
    df.loc[:,"N"] = df["value"].values
    "precipitation intensity (mm/h)"
    df.loc[:,"Ni"] = df["N"] / dt * 3600

    return df, dt


def create_dir(resmap, L):
    """ create directory for output to which results are written to

    Parameters
    ----------
        'resmap' (str): name/path of main output directory

    Returns
    -------
        'L' (list): list of names which have to be written under res directory
    """

    for i in range(len(L)):
        if not os.path.exists(os.path.join(resmap, L[i])):
            os.makedirs(os.path.join(resmap, L[i]))


if __name__ == "__main__":

    fname = "input.csv"
    dformat = "%Y-%m-%d %H:%M:%S"
    full_write = True,
    delimiter = ","

    ComputeRainfallErosivity(fname,dformat=dformat,full_write=full_write,delimiter=delimiter)