import numpy as np


def generate_square_network_with_crossbar():
    node_positions = np.array([[0,0],[1,0],[0,1],[1,1]])
    edges = np.array([[0,1,1,1],[1,0,0,1],[1,0,0,1],[1,1,1,0]])
    return node_positions, edges
    
    
def generate_bone_like_network(Ny, Number_small_filaments, Lx, Ly):
    large_rod_positions = np.zeros((Ny*Number_small_filaments, 2))
    for i in range(Ny):
        Lx_list = np.linspace(0, Lx, Number_small_filaments)
        #Lx_list = np.random.uniform(0, Lx, Number_small_filaments)
        Lx_list = np.sort(Lx_list)
        for j in range(Number_small_filaments):
            large_rod_positions[Number_small_filaments*i + j] = [Lx_list[j], i*Ly]
        #large_rod_end_positions[2*i] = [0, i*Ly]
        #large_rod_end_positions[2*i + 1] = [Lx, i*Ly]

    total_nodes = len(large_rod_positions)
    adjacency_matrix = np.zeros((total_nodes, total_nodes))

    for ii in range(total_nodes):
        for jj in range(ii+1, total_nodes):
            #find slope of line joining large_rod_positions[ii] and large_rod_positions[jj]
            if abs(large_rod_positions[ii][1]- large_rod_positions[jj][1])<Ly*1e-5:
                adjacency_matrix[ii][jj] = 1
                adjacency_matrix[jj][ii] = 1
            break

    for i in range(Ny-1):
        #get the index of node poistions in the i-th row
        indices_i = np.where(large_rod_positions[:,1] == i*Ly)
        indices_i_plus1 = np.where(large_rod_positions[:,1] == (i+1)*Ly)
        #randomly pair up the nodes in the i-th row with the nodes in the i+1-th row
        #i_random = np.random.choice(indices_i[0],Number_small_filaments, replace=False)
        #i_plus1_random = np.random.choice(indices_i_plus1[0],Number_small_filaments, replace=False)
        for j in range(Number_small_filaments-1):
            adjacency_matrix[indices_i[0][j]][indices_i_plus1[0][j+1]] = 1
            adjacency_matrix[indices_i_plus1[0][j+1]][indices_i[0][j]] = 1

            adjacency_matrix[indices_i[0][j+1]][indices_i_plus1[0][j]] = 1
            adjacency_matrix[indices_i_plus1[0][j]][indices_i[0][j+1]] = 1
    return large_rod_positions, adjacency_matrix