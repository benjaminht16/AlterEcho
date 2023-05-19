import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from models.diffusion.advanced_diffusion_modeling.hardi_analysis import compute_mean_diffusivity
from models.diffusion.diffusion_connectivity.connectivity_matrix import compute_connectivity_matrix

def plot_results(results_file, diffusion_data_file, connectivity_matrix_file):
    
    """
    Plot the results of a diffusion imaging analysis.
     Parameters:
    - results_file (str): Path to file containing the results of the analysis.
    - diffusion_data_file (str): Path to file containing the diffusion data.
    - connectivity_matrix_file (str): Path to file containing the connectivity matrix.
     Returns:
    - None
    """
    
     # Load results and data
    results = np.load(results_file)
    diffusion_data = nib.load(diffusion_data_file).get_fdata()
    connectivity_matrix = np.load(connectivity_matrix_file)
    
     # Compute mean diffusivity
    mean_diffusivity = compute_mean_diffusivity(diffusion_data)
    
     # Compute average connectivity
    average_connectivity = np.mean(connectivity_matrix)
    
     # PCA analysis on connectivity matrix
    pca = PCA(n_components=3)
    pca_matrix = pca.fit_transform(connectivity_matrix)
    
    # Train a neural network to predict disease progression    
    # Load data and labels
    data = np.load('data.npy')
    labels = np.load('labels.npy')

    # Split data into training and testing sets
    train_data = data[:500]
    train_labels = labels[:500]
    test_data = data[500:]
    test_labels = labels[500:]

    # Define and compile the model
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(train_data, train_labels, epochs=10, batch_size=32, validation_data=(test_data, test_labels))

    # Evaluate the model
    predictions = model.predict(test_data)
    accuracy = accuracy_score(test_labels, predictions.round())

     # Plot results
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
     axs[0].imshow(mean_diffusivity[:, :, 20], cmap='gray')
    axs[0].set_title('Mean diffusivity')
     axs[1].scatter(pca_matrix[:, 0], pca_matrix[:, 1], c=labels, cmap='coolwarm')
    axs[1].set_title('Connectivity matrix PCA')
     axs[2].bar(['Accuracy'], [accuracy])
    axs[2].set_title('Neural network accuracy')
     plt.show()