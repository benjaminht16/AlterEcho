import numpy as np 
import matplotlib.pyplot as plt
import networkx as nx
from dipy.core.gradients import gradient_table
from dipy.core.sphere import Sphere
from dipy.reconst.dti import TensorModel, fractional_anisotropy, color_fa
from dipy.reconst.dki import DiffusionKurtosisModel
from dipy.reconst.dki_micro import dki_micro
from dipy.reconst.shm import SphHarmModel 
from mpl_toolkits.mplot3d import Axes3D

def visualize_diffusion_perfusion(diffusion_map, perfusion_map, fig_size=(10, 5)):
    fig, axs = plt.subplots(1, 2, figsize=fig_size)
    axs[0].imshow(diffusion_map, cmap='gray', interpolation='none')
    axs[0].set_title('Diffusion Map')
    axs[1].imshow(perfusion_map, cmap='jet', interpolation='none')
    axs[1].set_title('Perfusion Map')
    plt.show()
    return fig
   
def visualize_diffusion_relaxometry_with_b0_and_bvecs(data, bvalues, bvecs):
    num_shells = len(bvalues)
     # Separate b0 and diffusion shells
    b0_shell = data[:, :, :, np.argwhere(bvalues == 0)[0][0]]
    shells = data[:, :, :, np.argwhere(bvalues != 0).ravel()]
    bvecs = bvecs[bvalues != 0]
    bvalues = bvalues[bvalues != 0]
     # Display diffusion shells
    fig = plt.figure(figsize=(15, 8))
    for i, (shell, bval, bvec) in enumerate(zip(shells.transpose((3, 0, 1, 2)), bvalues, bvecs)):
        ax = fig.add_subplot(2, num_shells, i+1, projection='3d')
        ax.voxels(shell > np.percentile(shell, 50), edgecolor='gray', alpha=0.1)
        ax.set_title(f"b-value: {bval}\n b-vector: {bvec}")
     # Display b0 shell
    ax = fig.add_subplot(2, num_shells, num_shells+1)
    ax.imshow(b0_shell[:, :, 0], cmap='gray')
    ax.set_title("b0 shell")
    plt.show()



def visualize_diffusion_microscopy(diffusion_map, fluorescence_map, fig_size=(10, 5)):
    fig, axs = plt.subplots(1, 2, figsize=fig_size)
    axs[0].imshow(diffusion_map, cmap='gray', interpolation='none')
    axs[0].set_title('Diffusion Map')
    axs[1].imshow(fluorescence_map, cmap='jet', interpolation='none')
    axs[1].set_title('Fluorescence Map')
    plt.show()
    return fig
   
def visualize_diffusion_weighted_imaging_with_b0(data, bvalues, bvecs, figsize=(15, 8)):
    num_shells = len(bvalues)

    # Separate b0 and diffusion shells
    b0_shell = data[:, :, :, np.argwhere(bvalues == 0)[0][0]]
    shells = data[:, :, :, np.argwhere(bvalues != 0).ravel()]
    bvecs = bvecs[bvalues != 0]
    bvalues = bvalues[bvalues != 0]

    # Display diffusion shells
    fig = plt.figure(figsize=figsize)
    for i, (shell, bval, bvec) in enumerate(zip(shells.transpose((3, 0, 1, 2)), bvalues, bvecs)):
        ax = fig.add_subplot(2, num_shells, i+1, projection='3d')
        ax.voxels(shell > np.percentile(shell, 50), edgecolor='gray', alpha=0.1)
        ax.set_title(f"b-value: {bval}\n b-vector: {bvec}")

    # Display b0 shell
    ax = fig.add_subplot(2, num_shells, num_shells+1)
    ax.imshow(b0_shell[:, :, 0], cmap='gray')
    ax.set_title("b0 shell")

    plt.show()

def visualize_multi_shell_diffusion_weighted_imaging_with_b0_and_bvecs(data, bvalues, bvecs):
    b0_idx = np.where(bvalues == 0)[0][0] #find the index of the b0 shell
    b0_shell = data[..., b0_idx] #extract the b0 shell from the data
    shells = data[..., b0_idx+1:] #extract the diffusion shells from data
    bvalues = bvalues[b0_idx+1:] #extract the corresponding b-values
    bvecs = bvecs[b0_idx+1:] #extract the corresponding b-vectors
    num_shells = len(bvalues)
    
    # Display the b0 shell intensity map
    fig, axs = plt.subplots(1, num_shells+1, figsize=(16, 4))
    axs[0].imshow(b0_shell, cmap='gray', vmin=np.percentile(b0_shell, 5), vmax=np.percentile(b0_shell, 95))
    axs[0].set_title('b0 shell')

    # Iterate over remaining shells and display with bvecs
    for i, (shell, bval, bvec) in enumerate(zip(shells.transpose((2,0,1)), bvalues, bvecs)):
        axs[i+1].imshow(shell, cmap='gray', vmin=np.percentile(shell, 5), vmax=np.percentile(shell, 95))
        axs[i+1].set_title(f'b-value = {bval}')
        
        # Display b-vectors as arrows on top of the image
        ax = fig.add_subplot(1, num_shells+1, i+2, projection='3d')
        ax.voxels(shell > np.percentile(shell, 50), alpha=0.025, edgecolor='gray') #display voxel outline
        ax.set_title(f'b-vec = {bvec}')
        x, y, z = np.array([[[0, 0, 0], bvec]]).transpose(0, 2, 1)
        ax.quiver(x[:,0], x[:,1], x[:,2], y[:,0], y[:,1], y[:,2], color='r', length=1.5) #display arrow
    
    plt.show()
    return fig

def visualize_diffusion_connectivity_with_labels_and_colors_and_edge_weights_and_node_weights(data, labels, colors, threshold):
    # Threshold data
    data[data < threshold] = 0
    
    # Compute correlation matrix
    corr_mat = np.corrcoef(data)
    
    # Create graph from correlation matrix
    G = nx.from_numpy_array(corr_mat)
    
    # Add node labels and colors to the graph
    for i in range(len(labels)):
        G.nodes[i]['label'] = labels[i]
        G.nodes[i]['color'] = colors[i]
    
    # Add edge weights to the graph
    weights = corr_mat[np.triu_indices(len(labels), k=1)]
    for i, (u, v, w) in enumerate(G.edges(data=True)):
        w['weight'] = weights[i]
    
    # Add node weights to the graph
    node_weight = np.sum(np.abs(data), axis=0)
    node_weight_dict = {i: node_weight[i] for i in range(len(labels))}
    nx.set_node_attributes(G, node_weight_dict, 'weight')
    
    # Compute node positions with spring layout
    pos = nx.spring_layout(G)
    
    # Draw the graph with nodes colored by their respective color, labels and weights and edges weighted by their weight
    node_color = [G.nodes[i]['color'] for i in range(len(labels))]
    node_size = [degree * 100 for node, degree in G.degree()]
    node_weight = nx.get_node_attributes(G, 'weight')
    edge_width = [w['weight'] * 5 for u, v, w in G.edges(data=True)]
    
    nx.draw_networkx_nodes(G, pos, node_color=node_color, node_size=node_size, cmap=plt.cm.Blues, 
                           node_shape='s', linewidths=1, edgecolors='black', alpha=0.7, 
                           vmin=min(node_weight.values()), vmax=max(node_weight.values()), 
                           node_weights=list(node_weight.values()))
    
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=edge_width)
    
    # Show degree, color, weight and node labels
    labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_color='white')
    
    # Show node weights
    node_weight_str_dict = {i: "{:.2f}".format(node_weight[i]) for i in range(len(labels))}
    nx.draw_networkx_node_labels(G, pos, labels=node_weight_str_dict, font_size=8)
    
    # Show edge weights
    edge_labels = {(u, v): round(w['weight'], 2) for u, v, w in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, label_pos=0.5, font_size=7)
    
    plt.axis('off')
    plt.show()(data, labels, colors, threshold):

def visualize_advanced_diffusion_modeling(data, t2_map, dti_params, qiv_params, bvals, bvecs):
    
    # Display data
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(data[:, :, data.shape[2]//2, 0], cmap='gray')
    axs[0].set_title('DWI Image')
    axs[1].imshow(t2_map, cmap='gray')
    axs[1].set_title('T2 Map')
    axs[2].hist(data.flatten(), bins=100, range=(0, np.max(data)))
    axs[2].set_title('DWI Histogram')
    axs[2].set_xlabel('DWI Intensity')
    axs[2].set_ylabel('Count')

    # Preprocess data
    gtab = gradient_table(bvals, bvecs)
    dti_model = TensorModel(gtab)
    dk_model = DiffusionKurtosisModel(gtab)
    tensor_fit = dti_model.fit(data)
    dki_fit = dk_model.fit(data)

    # Display DTI parameters
    md = tensor_fit.md
    fa = tensor_fit.fa
    axs[0].contour(md, colors='red', linewidths=.5)
    axs[0].contour(fa, colors='green', linewidths=.5)
    axs[0].set_title('DTI Parameters')

    # Display QIV parameters
    if qiv_params is not None:
        dki_m = dki_micro(dki_fit, data, qiv_params).fit()
        kurt = dki_m.kurtosis_tensor
        axs[1].contour(kurt[..., 0], colors='red', linewidths=.5)
        axs[1].contour(kurt[..., 1], colors='green', linewidths=.5)
        axs[1].contour(kurt[..., 2], colors='blue', linewidths=.5)
        axs[1].set_title('QIV Parameters')

    # Display DTI residuals
    res = tensor_fit.predict(gtab) - data
    axs[2].hist(res.flatten(), bins=100, range=(-np.max(data)/10, np.max(data)/10))
    axs[2].set_title('DTI Residuals')

    plt.show()
    return fig

def visualize_spectrum_imaging(data, bvals, bvecs, sh_order=8, sphere_samples=100):
    # Display data
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(data[:, :, data.shape[2]//2], cmap='gray')
    axs[0].set_title('DWI Image')
    axs[1].hist(data[:, :, data.shape[2]//2].flatten(), bins=100, range=(0, np.max(data)))
    axs[1].set_title('DWI Histogram')
    axs[1].set_xlabel('DWI Intensity')
    axs[1].set_ylabel('Count')
    # Preprocess data
    gtab = gradient_table(bvals, bvecs)
    sphere = Sphere(xyz=np.random.randn(sphere_samples, 3))
    sh_model = SphHarmModel(gtab, sh_order=sh_order, assume_normed=True)
    sh_coeffs = sh_model.fit(data).shm_coeff
    # Display SH coefficients
    sh_coeffs_img = np.zeros(sphere.vertices.shape[0])
    sh_coeffs_img[sh_model.sh_inv] = sh_coeffs
    axs[0].scatter(sphere.vertices[:, 0], sphere.vertices[:, 1], c=sh_coeffs_img, cmap='jet', s=10)
    axs[0].set_title('SH Coefficients')
    # Display SH reconstruction
    odf = sh_model.odf(sphere, sh_coeffs)
    axs[1].imshow(odf.reshape((sphere_samples, sphere_samples)), cmap='gray', interpolation='none')
    axs[1].set_title('ODF')
    plt.show()
    return fig

def visualize_diffusion_tensor_imaging(dwi_path, bval_path, bvec_path, mask_path=None, figsize=(15, 5)):
    # Load data and create gradient table
    dwi_data = load_nifti_data(dwi_path)
    bvals = np.loadtxt(bval_path)
    bvecs = np.loadtxt(bvec_path).T
    gtab = gradient_table(bvals, bvecs)

    # Compute diffusion tensor
    tenmodel = TensorModel(gtab)
    tenfit = tenmodel.fit(dwi_data)

    # Compute FA and color FA
    fa = fractional_anisotropy(tenfit.evals)
    cfa = color_fa(tenfit.fa, tenfit.evecs)

    # Apply mask if provided
    if mask_path is not None:
        mask_data = load_nifti_data(mask_path)
        fa = fa * mask_data
        cfa = cfa * np.tile(mask_data[..., None], (1, 1, 1, 3))

    # Display FA and color FA maps
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    axs[0].imshow(np.rot90(fa[:, fa.shape[1] // 2, :]), cmap='gray', interpolation='none')
    axs[0].set_title('Fractional Anisotropy')
    axs[1].imshow(np.rot90(cfa[:, cfa.shape[1] // 2, :, :]), interpolation='none')
    axs[1].set_title('Color FA')
    plt.show()
