import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from models.diffusion.diffusion_tensor_imaging import dti_analysis
from models.diffusion.diffusion_spectrum_imaging import dsi_analysis
from models.diffusion.advanced_diffusion_modeling import qball_analysis, hardi_analysis, dki_analysis
from models.diffusion.advanced_diffusion_modeling.diffusion_connectivity import connectivity_matrix, network_analysis
from alterecho.neural_interface import AlterEchoInterface
from alterecho.output_parser import parse_output

def plot_dti_eigenvectors(eigenvectors):
    """
    Plot the three principal eigenvectors of a DTI tensor.
    """
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    for i in range(3):
        ax[i].imshow(eigenvectors[..., i])
        ax[i].set_title(f"Principal eigenvector {i+1}")
    plt.show()


def plot_dsi_odf(odf):
     """
     Plot the orientation distribution function (ODF) of a diffusion spectrum imaging (DSI) dataset.
     """
     fig, ax = plt.subplots(figsize=(6, 6))
     ax = sns.histplot(odf.ravel(), bins=50, ax=ax)
     ax.set_xlabel('ODF amplitude')
     ax.set_ylabel('Count')
     plt.show()
    

def plot_qball_odf(odf):
    """
    Plot the orientation distribution function (ODF) of a q-ball imaging dataset.
    """
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = np.mgrid[-10:10:50j, -10:10:50j, -10:10:50j]
    ax.voxels(np.abs(odf), edgecolor='k')
    plt.show()


def plot_hardi_fiber_orientation(fiber_orientation):
    """
    Plot the reconstructed fiber orientation distribution of a high-angular-resolution diffusion imaging (HARDI)
    dataset.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax = sns.histplot(fiber_orientation.ravel(), bins=50, ax=ax)
    ax.set_xlabel('Fiber orientation')
    ax.set_ylabel('Count')
    plt.show()
    
    
def plot_dki_kurtosis(kurtosis):
    """
    Plot the kurtosis tensor of a diffusion kurtosis imaging (DKI) dataset.
    """
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    for i in range(3):
        ax[i].imshow(kurtosis[..., i])
        ax[i].set_title(f"Kurtosis tensor {i+1}")
    plt.show()
    
    
def plot_connectivity_matrix(mat):
    """
    Plot the connectivity matrix of a white matter fiber tractography dataset.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax = sns.heatmap(mat, cmap='coolwarm', square=True, cbar_kws={"shrink": .5}, ax=ax)
    ax.set_xlabel('Brain region')
    ax.set_ylabel('Brain region')
    plt.show()
    
    
def plot_network_properties(metrics):
    """
    Plot the network measures of a white matter fiber tractography dataset.
    """
    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(121)
    ax1 = sns.histplot(metrics['degree'], bins=50, kde=False, ax=ax1)
    ax1.set_xlabel('Degree')
    ax1.set_ylabel('Count')

    ax2 = fig.add_subplot(122)
    ax2 = sns.scatterplot(x=metrics['degree'], y=metrics['clustering'], hue=metrics['betweenness'], s=100, ax=ax2)
    ax2.set_xlabel('Degree')
    ax2.set_ylabel('Clustering coefficient')
    plt.show()


if __name__ == "__main__":
    # Initialize AlterEcho interface
    interface = AlterEchoInterface()

    # Simulate user input from AlterEcho system
    user_input = "Display DTI eigenvectors"

    # Parse user input and determine action
    action = parse_output(user_input)

    # Load the appropriate dataset based on the parsed action
    if action == "dti_eigenvectors":
        dti_data = np.load("data/dti_dataset.npy")
        dti_tensors = dti_analysis.fit_dti(dti_data)
        plot_dti_eigenvectors(dti_tensors['eigenvectors'])
    elif action == "dsi_ODF":
        dsi_data = np.load("data/dsi_dataset.npy")
        ODF = dsi_analysis.fit_dsi(dsi_data)
        plot_ODF(ODF)
    elif action == "qball_fODF":
        qball_data = np.load("data/qball_dataset.npy")
        fODF = qball_analysis.fit_qball(qball_data)
        plot_fODF(fODF)
    elif action == "hardi_fiber_tracking":
        hardi_data = np.load("data/hardi_dataset.npy")
        fiber_tracks = hardi_analysis.track_fibers(hardi_data)
        plot_fiber_tracks(fiber_tracks)
    elif action == "dki_kurtosis":
        dki_data = np.load("data/dki_dataset.npy")
        kurtosis_tensors = dki_analysis.fit_dki(dki_data)
        plot_kurtosis(kurtosis_tensors['kurtosis'])
    elif action == "connectivity_matrix":
        connectivity_data = np.load("data/connectivity_dataset.npy")
        connectivity_mat = connectivity_matrix.compute_connectivity_matrix(connectivity_data)
        plot_connectivity_matrix(connectivity_mat)
    elif action == "network_analysis":
        connectivity_data = np.load("data/connectivity_dataset.npy")
        connectivity_mat = connectivity_matrix.compute_connectivity_matrix(connectivity_data)
        network = network_analysis.analyze_network(connectivity_mat)
        plot_network(network)
    else:
        print("Invalid action specified. Please try again.")