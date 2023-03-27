from scipy.spatial import Delaunay
from scipy.spatial.qhull import QhullError

import itertools
import argparse
import numpy as np
import os
import pathlib
from tqdm import tqdm

from typing import Tuple


dataset_cls = ['brandenburg_gate', 'buckingham_palace', 'colosseum_exterior',
                  'grand_place_brussels', 'hagia_sophia_interior', 'notre_dame_front_facade',
                  'palace_of_westminster', 'pantheon_exterior', 'prague_old_town_square',
                  'taj_mahal', 'temple_nara_japan', 'trevi_fountain', 'westminster_abbey',
                   'reichstag', 'sacre_coeur', 'st_peters_square']


def extract_point_and_build(annotation_path, stg):
    for cls in dataset_cls:
        cls_path = os.path.join(annotation_path, cls)
        zero_imgs = []
        for file in tqdm(list(pathlib.Path(cls_path).glob('*.npz'))):
            file_name = str(file)
            pure_file_name = file_name.split('/')[-1]
            if pure_file_name[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                f = np.load(file)
                ori_point_set = f['points']
                point_set = f['points'][1:].transpose()
                num_points = point_set.shape[0]
                if num_points == 0:
                    zero_imgs.append(pure_file_name)
                if num_points < 2:
                    continue
                A, edge_num = build_graphs(point_set, num_points, stg=stg)
                np.savez(file_name, points=ori_point_set, adjacency_mat=A)
        zero_imgs = np.array(zero_imgs, dtype='<U24')
        zero_img_file = os.path.join(cls_path, 'zero_point_img.npz')
        np.savez(zero_img_file, img_name=zero_imgs)


def build_graphs(P_np: np.ndarray, n: int, n_pad: int=None, edge_pad: int=None, stg: str='fc',
                 thre: int=0) -> Tuple[np.ndarray, int]:

    assert stg in ('fc', 'tri', 'near'), 'No strategy named {} found.'.format(stg)

    if stg == 'tri':
        A = np.zeros((P_np.shape[0], P_np.shape[0]))
        A = delaunay_triangulate(P_np[0:n, :])
    elif stg == 'near':
        A = fully_connect(P_np[0:n, :], thre=thre)
    else:
        A = fully_connect(P_np[0:n, :])
    edge_num = int(np.sum(A, axis=(0, 1)))
    assert n > 0 and edge_num > 0, 'Error in n = {} and edge_num = {}'.format(n, edge_num)

    if n_pad is None:
        n_pad = n
    if edge_pad is None:
        edge_pad = edge_num
    assert n_pad >= n
    assert edge_pad >= edge_num

    return A, edge_num


def delaunay_triangulate(P: np.ndarray) -> np.ndarray:
    r"""
    Perform delaunay triangulation on point set P.

    :param P: :math:`(n\times 2)` point set
    :return: adjacency matrix :math:`A`
    """
    n = P.shape[0]
    if n < 3:
        A = fully_connect(P)
    else:
        try:
            d = Delaunay(P)
            A = np.zeros((n, n))
            for simplex in d.simplices:
                for pair in itertools.permutations(simplex, 2):
                    A[pair] = 1
        except QhullError as err:
            print('Delaunay triangulation error detected. Return fully-connected graph.')
            print('Traceback:')
            print(err)
            A = fully_connect(P)
    return A


def fully_connect(P: np.ndarray, thre=None) -> np.ndarray:
    r"""
    Return the adjacency matrix of a fully-connected graph.

    :param P: :math:`(n\times 2)` point set
    :param thre: edges that are longer than this threshold will be removed
    :return: adjacency matrix :math:`A`
    """
    n = P.shape[0]
    A = np.ones((n, n)) - np.eye(n)
    if thre is not None:
        for i in range(n):
            for j in range(i):
                if np.linalg.norm(P[i] - P[j]) > thre:
                    A[i, j] = 0
                    A[j, i] = 0
    return A


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno_path', dest='anno_path', help='dataset annotation directory',
                        default='data/IMC-PT-SparseGM/annotations',
                        type=str)
    parser.add_argument('--stg', dest='strategy', help='strategy of graph building, tri or near or fc',
                        default='tri',
                        type=str)
    args = parser.parse_args()

    extract_point_and_build(args.anno_path, args.stg)
