import matplotlib.pyplot as plt
import numpy as np
import os
def change_directory(path):
    path = input("Enter the path to the directory you want to change to: ")
    if os.path.isdir(path):  # Check if the entered path is a directory
        os.chdir(path)  # Change the current working directory to the specified path
        print(f"Changed directory to: {os.getcwd()}")
    else:
        print("Invalid path. Please enter a valid directory path.")
              


def load_data(filename):
    return np.loadtxt(filename)


def draw_spring(x0, y0, x1, y1, ax, num_coils=20):
    dx = x1 - x0
    dy = y1 - y0
    d = np.hypot(dx, dy)

    if d > 0:
        nx = dx / d
        ny = dy / d
        #dd = np.linspace(0, d, round(d/spring_length_factor))
        #dz = dd / d * np.pi * 2.0
        t = np.linspace(0, d, num_coils * 100)
        #x = x0 + nx * t
        #y = y0 + 0.2 * np.cos(t)
        x = x0 + nx * t + 2 * np.sin(t * num_coils * 2 * np.pi / d) * d / (num_coils * 20)
        y = y0 + ny * t - 2 * np.cos(t * num_coils * 2 * np.pi / d) * d / (num_coils * 20)
        # x = x0 + nx * dd + np.sin(dz) * spring_width * ny
        # y = y0 + ny * dd - np.sin(dz) * spring_width * nx
        ax.plot(x, y, color='black')
    return ax



def plot_network(node_positions, edges, data_folder,spring=False,special_points=None, filename='network'):
    fig, ax = plt.subplots()
    ax.scatter(node_positions[:, 0], node_positions[:, 1], color='blue')
    if special_points is not None:
        colors = ['red', 'magenta', 'yellow', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan', 'green']
        for point in special_points:
            ax.scatter(point[0], point[1], color=colors.pop(0))
    for i in range(len(node_positions)):
        for j in range(i+1, len(node_positions)):
            if edges[i, j] == 1:
                #lw = int(ym[i,j])/5
                lw=1
                if spring:
                    ax = draw_spring(node_positions[i, 0], node_positions[i, 1], node_positions[j, 0], node_positions[j, 1], ax)
                else:
                    ax.plot([node_positions[i, 0], node_positions[j, 0]],[node_positions[i, 1], node_positions[j, 1]], color='black', lw = lw )
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    #don't show the axes and ticks
    ax.axis('off')

    #plt.title('Triangular Network')
    #plt.gca().set_aspect('equal', adjustable='box')
    #plt.grid(True)
    # plt.show()

    plt.savefig(data_folder+filename+'.png', dpi=400)
    plt.savefig(data_folder+filename+'.svg', dpi=600)
    return fig, ax

def plot_network_from_folder(data_folder,spring=False,special_points=None):
    node_positions = load_data(data_folder+"node_positions.txt")
    edges = load_data(data_folder+"edges.txt")
    fig, ax = plt.subplots()
    ax.scatter(node_positions[:, 0], node_positions[:, 1], color='blue')
    if special_points is not None:
        colors = ['red', 'magenta', 'yellow', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan', 'green']
        for point in special_points:
            ax.scatter(point[0], point[1], color=colors.pop(0))
    for i in range(len(node_positions)):
        for j in range(i+1, len(node_positions)):
            if edges[i, j] == 1:
                #lw = int(ym[i,j])/5
                lw=1
                if spring:
                    ax = draw_spring(node_positions[i, 0], node_positions[i, 1], node_positions[j, 0], node_positions[j, 1], ax)
                else:
                    ax.plot([node_positions[i, 0], node_positions[j, 0]],[node_positions[i, 1], node_positions[j, 1]], color='black', lw = lw )
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    #plt.title('Triangular Network')
    #plt.gca().set_aspect('equal', adjustable='box')
    #plt.grid(True)
    #plt.show()
    plt.savefig(data_folder+'network.png', dpi=400)
    plt.savefig(data_folder+'network.svg', dpi=600)
    return fig, ax


    
if __name__ == "__main__":
    path = input("Enter the path to the directory you want to change to: ")
    change_directory(path)

    node_positions = load_data("node_positions.txt")
    edges = load_data("edges.txt")

    print(node_positions[:, 0])
    print(node_positions[:, 1])

    plot_network(node_positions, edges)