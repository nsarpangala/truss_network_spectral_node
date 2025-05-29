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
#tau_1,caps_lam_1, caps_lam_2=1002,1,1
#tau_2 = 1
#scale = 1 #to scale determigitnant
#pulse = np.sin(4 *0.04* np.pi * time)  # Create a sine wave

# laplacian_methods = {
#     "Springs": laplacian_balls_and_springs,
#     "Trusses": laplacian_pure_elastic
# }

def run_simulation(simname,w):
    data_folder = os.path.join(cwd, "data",simname)
    #create_or_replace_directory(data_folder)
    source_file = os.path.basename(__file__)
    # Define the destination file
    #destination_file = os.path.join(data_folder, source_file)
    #shutil.copyfile(source_file, destination_file)
    
    node_positions, edges = generate_square_network_with_crossbar()
    np.savetxt(data_folder+'edges.txt',edges)
    np.savetxt(data_folder+'node_positions.txt', node_positions)
    
    
    bc = [2, 3, 5]
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
    np.savetxt(data_folder+'Node_1_response_rectangle3'+str(w)+'.txt', Node_1_response)
    np.savetxt(data_folder+'Node_2_response_rectangle3'+str(w)+'.txt', Node_2_response)
    return Node_1_response, Node_2_response


simname = 'UPP1H29_vert_lam2_cmass/'
data_folder = os.path.join(cwd, "data",simname)
create_or_replace_directory(data_folder)

w_list = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000 ]

norm = plt.Normalize(0, 11)

# Get the coolwarm colormap
cmap = plt.cm.viridis

# Map the normalized values to the colormap
colors = cmap(norm(w_list))
def mech(u1, w, caps_lam_2_list, tau_2):
    return np.abs(u1)*(w*caps_lam_2_list*tau_2)

plt.close()

for ii in range(len(w_list)):
    w = w_list[ii]
    Node_1_response, Node_2_response =  run_simulation(simname, w)

    plt.plot(caps_lam_2_list, np.abs(Node_2_response),label=str(w))

plt.plot(caps_lam_2_list, np.abs(Node_1_response), color='k', label='Node 1')
plt.axvline(x=caps_lam_1, ls='--', color='k')
plt.xlabel(r'$\Lambda_2$', fontsize=20)

plt.ylabel('Response', fontsize=20)
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.tight_layout()
plt.savefig('data/'+simname+'response.png', dpi = 300)
plt.savefig('data/'+simname+'response.svg', dpi = 300)
plt.show()