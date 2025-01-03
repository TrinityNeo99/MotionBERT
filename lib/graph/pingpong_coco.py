#  Copyright (c) 2023. IPCRC, Lab. Jiangnig Wei
#  All rights reserved

import sys
sys.path.append("../../")
from lib.graph import graph_tools as tools

num_node = 17
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [(1, 3), (1, 0), (2, 4), (2, 0), (0, 5), (0, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11),
                    (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (5, 6)]
inward = [(i, j) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class AdjMatrixGraph:
    def __init__(self, *args, **kwargs):
        self.edges = neighbor
        self.num_nodes = num_node
        self.self_loops = [(i, i) for i in range(self.num_nodes)]
        self.A_binary = tools.get_adjacency_matrix(self.edges, self.num_nodes)
        self.A_binary_with_I = tools.get_adjacency_matrix(self.edges + self.self_loops, self.num_nodes)
        self.A = tools.normalize_adjacency_matrix(self.A_binary)
        self.A_binary_2 = self.A_binary @ self.A_binary
        self.A_binary_3 = self.A_binary @ self.A_binary @ self.A_binary
        self.A_binary_with_I_2 = self.A_binary_with_I @ self.A_binary_with_I
        self.A_binary_with_I_3 = self.A_binary_with_I @ self.A_binary_with_I @ self.A_binary_with_I
        self.A_binary_with_I_4 = self.A_binary_with_I_3 @ self.A_binary_with_I
        self.A_binary_with_I_5 = self.A_binary_with_I_4 @ self.A_binary_with_I
        self.A_binary_with_I_6 = self.A_binary_with_I_5 @ self.A_binary_with_I

        self.A_binary_with_I_2_mask = self.get_mask(self.A_binary_with_I_2)
        self.A_binary_with_I_3_mask = self.get_mask(self.A_binary_with_I_3)
        self.A_binary_with_I_4_mask = self.get_mask(self.A_binary_with_I_4)
        self.A_binary_with_I_5_mask = self.get_mask(self.A_binary_with_I_5)
        self.A_binary_with_I_6_mask = self.get_mask(self.A_binary_with_I_6)

    def get_mask(self, matrix):
        h, w = matrix.shape
        for i in range(h):
            for j in range(w):
                if matrix[i][j] != 0:
                    matrix[i][j] = 1
        return matrix

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    graph = AdjMatrixGraph()
    A, A_binary, A_binary_with_I = graph.A, graph.A_binary, graph.A_binary_with_I
    f, ax = plt.subplots(3, 3)
    ax[0, 0].imshow(A_binary_with_I, cmap='gray')
    ax[0, 1].imshow(A_binary, cmap='gray')
    ax[0, 2].imshow(A, cmap='gray')

    ax[1, 0].imshow(A_binary_with_I, cmap='gray')
    ax[1, 1].imshow(graph.A_binary_with_I_2_mask, cmap='gray')
    ax[1, 2].imshow(graph.A_binary_with_I_3_mask, cmap='gray')

    ax[2, 0].imshow(graph.A_binary_with_I_4_mask, cmap='gray')
    ax[2, 1].imshow(graph.A_binary_with_I_5_mask, cmap='gray')
    ax[2, 2].imshow(graph.A_binary_with_I_6_mask, cmap='gray')
    plt.show()
    print(A_binary_with_I.shape, A_binary.shape, A.shape)
    print(graph.A_binary_with_I_3)
