"""Various resampling schemes for particle filters"""
import numpy as np
import numpy.random as npr


def multinomial_resampling(w, M):
    """
    Multinomial resampling
    """
    N = w.size
    return npr.choice(N, size=M, replace=True, p=w).astype(int)


def stratified_resampling(w, M):
    """
    Stratified resampling
    """
    bins = np.cumsum(w)
    ind = np.arange(M)
    u = (ind + npr.rand(M)) / M

    return np.digitize(u, bins).astype(int)


def systematic_resampling(w, M):
    """
    Systematic resampling
    """
    bins = np.cumsum(w)
    ind = np.arange(M)
    u = (ind + npr.rand(1)) / M

    return np.digitize(u, bins).astype(int)


def resample(w, M, scheme='multinomial'):
    if scheme == 'multinomial':
        return multinomial_resampling(w, M)
    elif scheme == 'stratified':
        return stratified_resampling(w, M)
    elif scheme == 'systematic':
        return systematic_resampling(w, M)
    else:
        print('Not a valid resampling scheme!')
