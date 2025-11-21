import pandas as pd
import numpy as np
import pint
import tabulate

nlg_N2_df = pd.read_csv('nlg_N_2_L_100.dat.txt', sep=r'\s+', header=None)
xy_N2_df = pd.read_csv('wezly_N_2_L_100.dat.txt', sep=r'\s+', header=None)

nlg_N4_df = pd.read_csv('nlg_N_4_L100.dat.txt', sep=r'\s+', header=None)
xy_N4_df = pd.read_csv('wezly_N_4_L100.dat.txt', sep=r'\s+', header=None)

nlg_N6_df = pd.read_csv('nlg_N_6_L100.dat.txt', sep=r'\s+', header=None)
xy_N6_df = pd.read_csv('wezly_N_6_L100.dat.txt', sep=r'\s+', header=None)

nlg_N2_arr = nlg_N2_df.to_numpy(dtype=int)
xy_N2_arr = xy_N2_df.to_numpy(dtype=float)

nlg_N2_df.columns = ['k', 'i', 'global']
xy_N2_df.columns = ['k', 'x', 'y']

nlg_N4_arr = nlg_N4_df.to_numpy(dtype=int)
xy_N4_arr = xy_N4_df.to_numpy(dtype=float)

nlg_N4_df.columns = ['k', 'i', 'global']
xy_N4_df.columns = ['k', 'x', 'y']

nlg_N6_arr = nlg_N6_df.to_numpy(dtype=int)
xy_N6_arr = xy_N6_df.to_numpy(dtype=float)

nlg_N6_df.columns = ['k', 'i', 'global']
xy_N6_df.columns = ['k', 'x', 'y']

nlg_arr = {
    2: nlg_N2_arr,
    4: nlg_N4_arr,
    6: nlg_N6_arr
}

xy_arr = {
    2: xy_N2_arr,
    4: xy_N4_arr,
    6: xy_N6_arr
}

xy_df = {
    2: xy_N2_df,
    4: xy_N4_df,
    6: xy_N6_df
}

ureg = pint.UnitRegistry() # Initializing unit registry 

m0 =  ureg.electron_mass
hbar = 1 * ureg.hbar
m = 0.067 * m0
# N = 2
# L = 100 * ureg.nm / 0.05292
# aa = L / (2 * N)
omega = 10 * ureg.meV / hbar / 27211.6

w_val = [
    (18 + np.sqrt(30)) / 36,
    (18 + np.sqrt(30)) / 36,
    (18 - np.sqrt(30)) / 36,
    (18 - np.sqrt(30)) / 36
]

p_val = [
    - np.sqrt(3/7 - 2/7 * np.sqrt(6/5)),
      np.sqrt(3/7 - 2/7 * np.sqrt(6/5)),
      np.sqrt(3/7 + 2/7 * np.sqrt(6/5)),
    - np.sqrt(3/7 + 2/7 * np.sqrt(6/5))
]


# def nlg(k, i):
#     k_mask = (nlg_df['k'] == (k + 1))
#     i_mask = (nlg_df['i'] == (i + 1))
#     gl = nlg_df[k_mask & i_mask]['global']
#     return gl.iloc[0]

def nlg(k, i, N):
    try:
        # if(N == 2):
            # print(f"wtf {k}, {i}, {N}")
        return nlg_arr[N][k * 9 + i, 2]
    except IndexError:
        raise Exception("Podano złą wartość N, musi być 2, 4 lub 6.")

# def xy(k, axis, L):
#     scale = L / 100
#     k_mask = (xy_df['k'] == (k + 1))
#     if axis == 'x':
#         return xy_df[k_mask]['x'].iloc[0] / .05292 * scale
#     if axis == 'y':
#         return xy_df[k_mask]['y'].iloc[0] /.05292 * scale
    
def xy(k, axis, L, N):
    if not(N == 2) and not(N == 4) and not(N == 6):
        raise Exception("Podano złą wartość N, musi być 2, 4 lub 6.")
    scale = L / 100
    if axis == 'x':
        return xy_arr[N][k, 1] / .05292 * scale
    if axis == 'y':
        return xy_arr[N][k, 2] /.05292 * scale

def f1(xi):
    return (1 - xi) / 2

def f2(xi):
    return (1 + xi) / 2

def g1(xi1, xi2):
    return f1(xi1) * f1(xi2)

def g2(xi1, xi2):
    return f1(xi2) * f2(xi1)

def g3(xi1, xi2):
    return f1(xi1) * f2(xi2)

def g4(xi1, xi2):
    return f2(xi1) * f2(xi2)

def q1(xi):
    return xi * (xi - 1) / 2

def q2(xi):
    return (1 - xi) * (1 + xi)

def q3(xi):
    return  xi * (xi + 1) / 2

def h1(xi1, xi2):
    return q1(xi1) * q1(xi2)

def h2(xi1, xi2):
    return q3(xi1) * q1(xi2)

def h3(xi1, xi2):
    return q1(xi1) * q3(xi2)

def h4(xi1, xi2):
    return q3(xi1) * q3(xi2)

def h5(xi1, xi2):
    return q2(xi1) * q1(xi2)

def h6(xi1, xi2): 
    return q3(xi1) * q2(xi2)

def h7(xi1, xi2):
    return q1(xi1) * q2(xi2)

def h8(xi1, xi2):
    return q2(xi1) * q3(xi2)

def h9(xi1, xi2):
    return q2(xi1) * q2(xi2)

h = [h1, h2, h3, h4, h5, h6, h7, h8, h9]

g = [g1, g2, g3, g4]


def dhdxi1(hi: callable, xi1, xi2):
    dxi = 1e-6 * xi1
    return ( hi(xi1 + dxi, xi2) - hi(xi1 - dxi, xi2) ) / (2 * dxi)

def dhdxi2(hi: callable, xi1, xi2):
    dxi = 1e-6 * xi2
    return ( hi(xi1, xi2 + dxi) - hi(xi1, xi2 - dxi) ) / (2 * dxi)

def dq1_dxi(xi):
    return (2 * xi - 1) / 2

def dq2_dxi(xi):
    return 2 * xi

def dq3_dxi(xi):
    return (2 * xi + 1) / 2

# h1 = q1(xi1) * q1(xi2)
def dh1_dxi1(xi1, xi2):
    return dq1_dxi(xi1) * q1(xi2)
def dh1_dxi2(xi1, xi2):
    return q1(xi1) * dq1_dxi(xi2)

# h2 = q3(xi1) * q1(xi2)
def dh2_dxi1(xi1, xi2):
    return dq3_dxi(xi1) * q1(xi2)
def dh2_dxi2(xi1, xi2):
    return q3(xi1) * dq1_dxi(xi2)

# h3 = q1(xi1) * q3(xi2)
def dh3_dxi1(xi1, xi2):
    return dq1_dxi(xi1) * q3(xi2)
def dh3_dxi2(xi1, xi2):
    return q1(xi1) * dq3_dxi(xi2)

# h4 = q3(xi1) * q3(xi2)
def dh4_dxi1(xi1, xi2):
    return dq3_dxi(xi1) * q3(xi2)
def dh4_dxi2(xi1, xi2):
    return q3(xi1) * dq3_dxi(xi2)

# h5 = q2(xi1) * q1(xi2)
def dh5_dxi1(xi1, xi2):
    return dq2_dxi(xi1) * q1(xi2)
def dh5_dxi2(xi1, xi2):
    return q2(xi1) * dq1_dxi(xi2)

# h6 = q3(xi1) * q2(xi2)
def dh6_dxi1(xi1, xi2):
    return dq3_dxi(xi1) * q2(xi2)
def dh6_dxi2(xi1, xi2):
    return q3(xi1) * dq2_dxi(xi2)

# h7 = q1(xi1) * q2(xi2)
def dh7_dxi1(xi1, xi2):
    return dq1_dxi(xi1) * q2(xi2)
def dh7_dxi2(xi1, xi2):
    return q1(xi1) * dq2_dxi(xi2)

# h8 = q2(xi1) * q3(xi2)
def dh8_dxi1(xi1, xi2):
    return dq2_dxi(xi1) * q3(xi2)
def dh8_dxi2(xi1, xi2):
    return q2(xi1) * dq3_dxi(xi2)

# h9 = q2(xi1) * q2(xi2)
def dh9_dxi1(xi1, xi2):
    return dq2_dxi(xi1) * q2(xi2)
def dh9_dxi2(xi1, xi2):
    return q2(xi1) * dq2_dxi(xi2)

dh_dxi1 = [dh1_dxi1, dh2_dxi1, dh3_dxi1, dh4_dxi1, dh5_dxi1, dh6_dxi1, dh7_dxi1, dh8_dxi1, dh9_dxi1]
dh_dxi2 = [dh1_dxi2, dh2_dxi2, dh3_dxi2, dh4_dxi2, dh5_dxi2, dh6_dxi2, dh7_dxi2, dh8_dxi2, dh9_dxi2]

def calcLocalOverlapMatrix(c1):
    lom = np.zeros((9, 9))
    for j in range(0, 9):
        for i in range(0, 9):
            lom[j][i] = calcLOMsingleElement(i, j)
    return c1 * lom
            

def calcLOMsingleElement(i, j):
    lom_ij = 0
    for l in range(0, 4):
        for n in range(0, 4):
            lom_ij += w_val[l] * w_val[n] * h[j](p_val[l], p_val[n]) * h[i](p_val[l], p_val[n])
    return lom_ij

# def calcKEMsingleElement(i, j):
#     dhi_dxi1 = dh_dxi1[i]
#     dhj_dxi1 = dh_dxi1[j]
#     dhi_dxi2 = dh_dxi2[i]
#     dhj_dxi2 = dh_dxi2[j]
#     kem_ij = 0
#     for l in range(0, 4):
#         for n in range(0, 4):
#             pl = p_val[l]
#             pn = p_val[n]
#             kem_ij += w_val[l] * w_val[n] * (dhi_dxi1(pl, pn) * dhj_dxi1(pl, pn) + dhi_dxi2(pl, pn) * dhj_dxi2(pl, pn))
    
#     return kem_ij

def calcKEMsingleElement(i, j):
    kem_ij = 0
    for l in range(0, 4):
        for n in range(0, 4):
            pl = p_val[l]
            pn = p_val[n]
            kem_ij += w_val[l] * w_val[n] * ( dhdxi1(h[i], pl, pn) * dhdxi1(h[j], pl, pn) + dhdxi2(h[i], pl, pn) * dhdxi2(h[j], pl, pn) )
    
    return kem_ij

# def calcKineticEnergyMatrix():
#     const = hbar **2 / (2 * m)
#     kem = np.zeros((9, 9))
#     for j in range(0, 9):
#         for i in range(0, 9):
#             kem[j][i] = calcKEMsingleElement(i, j)
#     return const * kem

def calcKineticEnergyMatrix():
    const = hbar **2 / (2 * m)
    kem = np.zeros((9, 9))
    for j in range(0, 9):
        for i in range(0, 9):
            kem[j][i] = calcKEMsingleElement(i, j)
    return const * kem

def calcX(xi1, xi2, k, L, N):
    x_ = 0
    for i in range(0, 4):
        globalIdx = nlg(k, i, N) - 1
        x_ += xy(globalIdx, 'x', L, N) * g[i](xi1, xi2)
    return x_

def calcY(xi1, xi2, k, L, N):
    y_ = 0
    for i in range(0, 4):
        globalIdx = nlg(k, i, N) - 1
        y_ += xy(globalIdx, 'y', L, N) * g[i](xi1, xi2)
    return y_

def calcPEMsingleElement(i, j, k, L, N):
    pem_ijk = 0
    for l in range(0, 4):
        for n in range(0, 4):
            wl = w_val[l]
            wn = w_val[n]
            pl = p_val[l]
            pn = p_val[n]
            x_ = calcX(pl, pn, k, L, N)
            y_ = calcY(pl, pn, k, L, N)
            pem_ijk += w_val[l] * w_val[n] * (x_**2 + y_**2) * h[j](pl, pn) * h[i](pl, pn)
    return pem_ijk

def calcLocalPotentialEnergyMatrix(k, L, const, N):
    pem_k = np.zeros((9, 9))
    for i in range(0, 9):
        for j in range(0, 9):
            pem_k[j][i] = calcPEMsingleElement(i, j, k, L, N)
    return const * pem_k

def buildPotentialEnergyMatrix(L, N):
    k_max = (2*N)**2
    pem = [0] * k_max
    for k in range(0, k_max):
        pem[k] =  calcLocalPotentialEnergyMatrix(k, L)
    return pem

def buildGlobalMatrices(N: int, L: float):
    """Generuje macierze S i H dla zadanych wartości N i L

    Args:
        :N (int): Zdefiniowane tak, że (2N)**2 to łączna liczba elementów w pudle obliczeniowym
        :L (float): Szerokość pudła w nm 

    Returns:
        :tuple: (S, H) gdzie:
        S (np.ndarray): Globalna macierz przekrywania.
        H (np.ndarray): Globalna macierz Hamiltona.
    """
    a = L / (2 * N) * ureg.nm / 0.05292

    c1 = a**2 / 4
    c2 = m * omega**2 / 2

    const = c1 * c2

    size = (4*N + 1)**2

    S = np.zeros((size, size))
    H = np.zeros((size, size))

    locS = calcLocalOverlapMatrix(c1).magnitude
    locT = calcKineticEnergyMatrix().magnitude

    for k in range(0, k_max := (2*N)**2):
        # update_progress(k, k_max - 1)
        locV = calcLocalPotentialEnergyMatrix(k, L, const, N).magnitude
        # if(k == 11):
            # print(tabulate.tabulate(locV, tablefmt="github", floatfmt=".3f"))
        for i in range(0, 9):
            for j in range(0, 9):
                i1 = nlg(k, i, N) - 1
                i2 = nlg(k, j, N) - 1
                S[i1, i2] += locS[i, j]
                H[i1, i2] += locT[i, j] + locV[i, j] 
    return S, H

def makeBoundCondsForGlobalMatrices(S: np.ndarray, H: np.ndarray):

    size = len(S)
    N = (np.sqrt(size) - 1) / 4

    if not(N == 2) and not(N == 4) and not(N == 6):
        raise Exception("Podano złą wartość N, musi być 2, 4 lub 6.")

    xMin = xy_df[N]['x'].min()
    xMax = xy_df[N]['x'].max()

    yMin = xy_df[N]['y'].min()
    yMax = xy_df[N]['y'].max()

    xMask = (xy_df[N]['x'] == xMin) | (xy_df[N]['x'] == xMax)
    yMask = (xy_df[N]['y'] == yMin) | (xy_df[N]['y'] == yMax)

    boundIndexes = xy_df[N][xMask | yMask]['k']
    boundIndexes -= 1
    boundIndexes.values
    bitwaPodGrunwaldem = 1410

    S[boundIndexes, :] = 0.0
    S[:, boundIndexes] = 0.0
    H[boundIndexes, :] = 0.0
    H[:, boundIndexes] = 0.0
    S[boundIndexes, boundIndexes] = 1.0
    H[boundIndexes, boundIndexes] = -bitwaPodGrunwaldem

    return None


from IPython.display import clear_output
import time
import sys


def update_progress(current, total, length=30):
    """
    Wyświetla pasek postępu w Jupyterze.
    current: aktualny krok (int)
    total: liczba wszystkich kroków (int)
    length: długość paska w znakach
    """
    progress = current / total
    filled = int(length * progress)
    bar = '█' * filled + '-' * (length - filled)
    percent = 100 * progress
    sys.stdout.write(f'\r[{bar}] {percent:6.2f}%')
    sys.stdout.flush()
    
    if current >= total:
        # po zakończeniu — usuń pasek
        clear_output(wait=True)