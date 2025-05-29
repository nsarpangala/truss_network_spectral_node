import sys
import os
sys.path.insert(0, os.path.join('..', 'truss_network_spectral_node', 'src'))
from network_laplacians import *
from network_generator import *
from data_management import *
from assign_network_parameters import *
#from heatmap_propagation_of_pulse import *
from analysis.visualize_net import *

import matplotlib.pyplot as plt

import shutil
import logging

# Set up logging
logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

cwd = os.getcwd()

#Define Parameters
dimensions = 2
L= 1
ratio_tau_lambda = 1
caps_lam_0 = 1
caps_lam_2_list = np.logspace(-7, 7, 100)
tau_2=1
tau_1 = 1
caps_lam_1 = 1
u1 =1
def run_simulation(simname,w):
    
    bc = [2, 3, 5]
    node_positions, edges = generate_square_network_with_crossbar()
    number_joints = len(node_positions)

#     #w_array, yf = compute_and_save_fft(pulse, s_rate, data_folder+'pulse')
    vect_P_til = np.zeros(2*number_joints)
    vect_P_til = np.delete(vect_P_til, bc)
    P2 = vect_P_til[1:]
    spring_data = {}
    truss_data = {}

    Node_1_response =[]
    Node_2_response =[]
    for caps_lam_2 in caps_lam_2_list:
        tau_2 = caps_lam_2

        unit_vector_matrix,tau_matrix, capital_Lambda_matrix, unit_vector_matrix_reshaped = assign_square_network_vert_lambda2(dimensions, number_joints, edges, node_positions, tau_1,tau_2,caps_lam_1, caps_lam_2)

        D_truss = laplacian_pure_elastic(dimensions, number_joints, edges,unit_vector_matrix, w, capital_Lambda_matrix, tau_matrix)
        D_truss = np.delete(D_truss, bc, axis=0)
        D_truss = np.delete(D_truss, bc, axis=1)
        A22 = D_truss[1:,1:]
        A21 = D_truss[1:,0]
        RHS = P2 - np.dot(A21,u1)
        U2 = np.linalg.solve(A22,RHS)
        u_til = np.insert(U2, 0, u1)
        #u_til = np. linalg.solve(D_truss, vect_P_til)
        for ii in range(len(bc)):
            u_til = np.insert(u_til, bc[ii], 0)
        Node_1_response.append(u_til[0])
        l = len(u_til)

        Node_2_response.append(u_til[l-1])
    return Node_1_response, Node_2_response


w_list = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000 ]

norm = plt.Normalize(0, 11)

def mech(u1, w, caps_lam_2_list, tau_2):
    return np.abs(u1)*(w*caps_lam_2_list*tau_2)

#node_1_response = np.loadtxt('tests/node1response.txt')
#node_2_response = np.loadtxt('tests/node2response.txt')
def test_program():
    for ii in range(len(w_list)):
        w = w_list[ii]
        Node_1_response, Node_2_response =  run_simulation('dum', w)
    #    np.savetxt('tests/test_data/node1response'+str(ii)+'.txt', Node_1_response)
    #    np.savetxt('tests/test_data/node2response'+str(ii)+'.txt', Node_2_response)
        node_1_response_ref = np.loadtxt('tests/test_data/node1response'+str(ii)+'.txt')
        node_2_response_ref = np.loadtxt('tests/test_data/node2response'+str(ii)+'.txt')
        
        
        
        assert np.allclose(node_1_response_ref,Node_1_response, rtol=1e-3, atol=1e-5), "Data Missmatch"
        assert np.allclose(node_2_response_ref,Node_2_response, rtol=1e-3, atol=1e-5), "Data Missmatch"

test_program()
