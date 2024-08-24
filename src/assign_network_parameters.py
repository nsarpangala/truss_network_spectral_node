import numpy as np

def assign_square_network_one_lambda2(dimensions, number_joints, edges, node_positions, tau_1,tau_2,caps_lam_1, caps_lam_2):
    unit_vector_matrix = np.zeros([number_joints, number_joints, dimensions])
    tau_matrix = np.zeros([number_joints, number_joints])
    capital_Lambda_matrix = np.zeros([number_joints, number_joints])    
    ii_values, jj_values = np.where(edges == 1)
    for inde in range(len(ii_values)):
        ii = ii_values[inde]
        jj = jj_values[inde]
        riijj = node_positions[jj] - node_positions[ii]
        length = np.sqrt(np.sum(riijj**2))   
        if abs(node_positions[jj][1] - node_positions[ii][1])<1e-5 or abs(node_positions[jj][0] - node_positions[ii][0])<1e-5:
            tau_matrix[ii, jj] = length*tau_1
            capital_Lambda_matrix[ii,jj] = caps_lam_1
        else:
            tau_matrix[ii, jj] = length*tau_2
            capital_Lambda_matrix[ii,jj] = caps_lam_2
        unit_vector_matrix[ii,jj] = riijj/np.sqrt(np.sum(riijj**2))    
    unit_vector_matrix_reshaped = unit_vector_matrix.reshape(unit_vector_matrix.shape[0], -1)
    return unit_vector_matrix,tau_matrix, capital_Lambda_matrix, unit_vector_matrix_reshaped

def assign_square_network_vert_lambda2(dimensions, number_joints, edges, node_positions, tau_1,tau_2,caps_lam_1, caps_lam_2):
    unit_vector_matrix = np.zeros([number_joints, number_joints, dimensions])
    tau_matrix = np.zeros([number_joints, number_joints])
    capital_Lambda_matrix = np.zeros([number_joints, number_joints])    
    ii_values, jj_values = np.where(edges == 1)
    for inde in range(len(ii_values)):
        ii = ii_values[inde]
        jj = jj_values[inde]
        riijj = node_positions[jj] - node_positions[ii]
        length = np.sqrt(np.sum(riijj**2))   
        if abs(node_positions[jj][1] - node_positions[ii][1])<1e-5:
            tau_matrix[ii, jj] = length*tau_1
            capital_Lambda_matrix[ii,jj] = caps_lam_1
        else:
            tau_matrix[ii, jj] = length*tau_2
            capital_Lambda_matrix[ii,jj] = caps_lam_2
        unit_vector_matrix[ii,jj] = riijj/np.sqrt(np.sum(riijj**2))    
    unit_vector_matrix_reshaped = unit_vector_matrix.reshape(unit_vector_matrix.shape[0], -1)
    return unit_vector_matrix,tau_matrix, capital_Lambda_matrix, unit_vector_matrix_reshaped


def assign_network_pure_elastic_non_dimensional(dimensions, number_joints, edges, node_positions, tau_0,caps_lam_0):
    unit_vector_matrix = np.zeros([number_joints, number_joints, dimensions])
    tau_matrix = np.zeros([number_joints, number_joints])
    capital_Lambda_matrix = np.zeros([number_joints, number_joints])    
    ii_values, jj_values = np.where(edges == 1)
    #print(youngs_mod_list)
    for inde in range(len(ii_values)):
        ii = ii_values[inde]
        jj = jj_values[inde]
        #print(edges[ii,jj])
        riijj = node_positions[jj] - node_positions[ii]
        length = np.sqrt(np.sum(riijj**2))
        
        tau_matrix[ii, jj] = length*tau_0
        unit_vector_matrix[ii,jj] = riijj/np.sqrt(np.sum(riijj**2))
        #print(node_positions[ii], node_positions[jj], unit_vector_matrix[ii,jj])
        capital_Lambda_matrix[ii,jj] = caps_lam_0
                
                
    unit_vector_matrix_reshaped = unit_vector_matrix.reshape(unit_vector_matrix.shape[0], -1)
    return unit_vector_matrix,tau_matrix, capital_Lambda_matrix, unit_vector_matrix_reshaped

