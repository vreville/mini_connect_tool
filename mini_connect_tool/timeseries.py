import numpy as np
import pandas as pd

def runningMean(x, N):
    return np.convolve(x, np.ones((N,))/N)[(N-1):]

def sign_change(a):
    asign = np.sign(a)
    signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)
    idx=np.where(signchange != 0)

    return idx

def interp(df, new_index):
    """Return a new DataFrame with all columns values interpolated to the new_index values."""
    df_out = pd.DataFrame(index=new_index)
    df_out.index.name = df.index.name

    for colname, col in df.items():
        df_out[colname] = np.interp(new_index, df.index, col)

    return df_out
