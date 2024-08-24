import numpy as np
import cmath
   
def laplacian_pure_elastic(dimensions, number_joints, edges,unit_vector_matrix, w, capital_Lambda, tau_matrix):
    '''dimensions, number_joints, edges,unit_vector_matrix, w, capital_Lambda, tau_matrix'''
    
    # Initialize the Laplacian matrix D with zeros
    D=np.zeros([dimensions*number_joints, dimensions*number_joints])
    epsilon = 1e-10 # Small value to avoid division by zero

    # Loop over each joint
    for mu in range(number_joints):
         # Find the connected nodes (edges) for the current joint
        gammas, = np.where(edges[mu,:]==1)
        for gamma in gammas:
            # Calculate the prefactor based on the frequency w and time constant tau_matrix
            if abs(w)>epsilon:
                prefactor = capital_Lambda[mu, gamma] * w / np.tan(w * tau_matrix[mu, gamma])
            else:
                prefactor = capital_Lambda[mu, gamma] / tau_matrix[mu, gamma]
            # Update the diagonal block of the Laplacian matrix D
            D[dimensions*mu:dimensions*mu+dimensions, dimensions*mu:dimensions*mu+dimensions] += prefactor*np.outer(unit_vector_matrix[mu, gamma], unit_vector_matrix[mu, gamma])
    
    # Find all pairs of connected nodes (edges)
    mues, nues = np.where(edges==1)
    for ide in range(len(mues)):
        mu = mues[ide]
        nu = nues[ide]
        # Calculate the prefactor for off diagonal elements based on the frequency w and time constant tau_matrix
        if abs(w)>epsilon:
            prefactor = capital_Lambda[mu, nu] * w / np.sin(w * tau_matrix[mu, nu])
        else:
            prefactor = capital_Lambda[mu, nu] / tau_matrix[mu, nu]
        # Update the off-diagonal block of the Laplacian matrix D
        D[dimensions* mu:dimensions * mu + dimensions, dimensions* nu:dimensions* nu + dimensions] = prefactor*np.outer(unit_vector_matrix[mu, nu], unit_vector_matrix[nu, mu])
    return D

def laplacian_balls_and_springs(dimensions, number_joints, edges, unit_vector_matrix, w, capital_Lambda, tau_matrix):
    '''dimensions, number_joints, edges, unit_vector_matrix, w, capital_Lambda, tau_matrix'''
    
    # Identity matrix of size 'dimensions'
    I = np.eye(dimensions)

     # Initialize the Laplacian matrix D with zeros
    D=np.zeros((dimensions*number_joints, dimensions*number_joints))

    # Loop over each joint
    for mu in range(number_joints):

         # Find the connected nodes (edges) for the current joint
        gammas, = np.where(edges[mu,:]==1)
        mass_of_connected_filaments = 0

        # Loop over each connected node
        for gamma in gammas:

            #stiffness of filament is capital_Lambda divided by tau
            prefactor = capital_Lambda[mu, gamma]/ tau_matrix[mu, gamma]
            D[dimensions*mu:dimensions*mu+dimensions, dimensions*mu:dimensions*mu+dimensions] += prefactor*np.outer(unit_vector_matrix[mu, gamma], unit_vector_matrix[mu, gamma])
            
            #mass of a filament is capital_Lambda times tau
            #add mass of all connected filaments at a joint
            mass_of_connected_filaments+=capital_Lambda[mu, gamma]*tau_matrix[mu, gamma]
        
        # Mass of a filament is shared equally between connecting nodes
        mass_of_connected_filaments = mass_of_connected_filaments/2

        # Update the diagonal block of the Laplacian matrix D with the mass term

        D[dimensions*mu:dimensions*mu+dimensions, dimensions*mu:dimensions*mu+dimensions] += (-1.0*mass_of_connected_filaments*w**2)*I
    
    # Find all pairs of connected nodes (edges)
    mues, nues = np.where(edges==1)

    # Loop over each pair of connected nodes
    for ide in range(len(mues)):
        mu = mues[ide]
        nu = nues[ide]

        # Calculate the prefactor for off diagonal elements based on the frequency w and time constant tau_matrix
        prefactor = capital_Lambda[mu, nu]/ tau_matrix[mu, nu]

        # Update the off-diagonal block of the Laplacian matrix D
        D[dimensions* mu:dimensions * mu + dimensions, dimensions* nu:dimensions* nu + dimensions] = prefactor*np.outer(unit_vector_matrix[mu, nu], unit_vector_matrix[nu, mu])
    return D


def test_laplacian(dimensions, number_joints, edges, capital_Lambda, w, density_matrix, A_matrix, tau_matrix, length_matrix, unit_vector_matrix):
    '''dimensions, capital_Lambda, w, number_joints, edges, tau_matrix, unit_vector_matrix'''
    m= 10
    mues, nues = np.where(edges==1)
    id = 0
    mu = mues[0]
    nu = nues[0]
    k=200
    #k = capital_Lambda[mu,nu]/tau_matrix[mu, nu]
    D= np.array([[-1.0*m*w**2+k,-1.0*k],[-1.0*k,-1.0*m*w**2+k]])
    return D

def laplacian_pure_elastic_single_filament(w, cL, tau):
    '''dimensions, number_joints, edges,unit_vector_matrix, w, capital_Lambda, tau'''
    if abs(w)>1e-8:
        cot=cL * w / np.tan(w * tau)
        cosex = cL * w / np.sin(w * tau)
        D = np.array([[cot, -cosex], [-cosex, cot]])
    else:
        D = np.array([[cL/tau, -cL/tau], [-cL/tau, cL/tau]])
    return D