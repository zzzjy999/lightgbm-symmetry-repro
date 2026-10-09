"""Function-generated data only; this module has no external-data reader."""
import argparse
from pathlib import Path
import numpy as np


def four_cells(a=1, repeats=128):
    """Balanced beta,z in {-1,+1}, y=beta*(a+z); columns are beta,z."""
    if a not in (0, 1) or repeats < 1:
        raise ValueError('a must be 0 or 1 and repeats must be positive')
    cells = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
    x = np.repeat(cells, repeats, axis=0)
    return x, x[:, 0] * (a + x[:, 1])


def noise_pair(y, seed=11):
    """Archived single-tree noise rule: IID SD=1e-4*SD(y), matched odd RMS."""
    if len(y) % 2:
        raise ValueError('paired data require an even number of rows')
    rng = np.random.default_rng(seed)
    iid = rng.normal(size=len(y)) * 1e-4 * np.std(y)
    half = rng.normal(size=len(y)//2)
    odd = np.concatenate([-half, half])
    odd *= np.sqrt(np.mean(iid**2) / np.mean(odd**2))
    return iid, odd


def continuous_shapes(n_shapes=256, seed=20261009):
    """New public validation data; beta in {-1,0,1}, five uniform z inputs.

    y = beta * (2 + 1.25 sin(pi*z0) + .7*z1**2 - 1.1*z2*z3 + .45*cos(2*z4)).
    Use different seeds for fitting and evaluation; mirrors share shape IDs.
    """
    if n_shapes < 1:
        raise ValueError('n_shapes must be positive')
    z = np.random.default_rng(seed).uniform(-1, 1, (n_shapes, 5))
    base = 2 + 1.25*np.sin(np.pi*z[:, 0]) + .7*z[:, 1]**2 - 1.1*z[:, 2]*z[:, 3] + .45*np.cos(2*z[:, 4])
    beta = np.tile([-1., 0., 1.], n_shapes)
    x = np.column_stack([beta, np.repeat(z, 3, axis=0)])
    return x, beta*np.repeat(base, 3), np.repeat(np.arange(n_shapes), 3)


def export(directory):
    directory.mkdir(parents=True, exist_ok=False)
    for a in (0, 1):
        x,y = four_cells(a)
        np.savetxt(directory/f'four_cells_a{a}.csv',np.column_stack([x,y]),delimiter=',',
                   header='beta,z,y',comments='',fmt='%.17g')
    for label, seed in [('train',20261009),('validation',20261010)]:
        x,y,shape = continuous_shapes(seed=seed)
        np.savetxt(directory/f'continuous_{label}.csv',np.column_stack([shape,x,y]),delimiter=',',
                   header='shape,beta,z0,z1,z2,z3,z4,y',comments='',fmt='%.17g')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    export(parser.parse_args().output)
