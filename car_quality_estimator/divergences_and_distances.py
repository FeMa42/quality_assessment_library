from scipy.stats import gaussian_kde
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from scipy.special import kl_div
import numpy as np

def compute_distribution_metrics(gen_distribution, ref_distribution):
    """
    Compute multiple distribution similarity metrics between generated and reference distributions.
    Args:
        gen_distribution (np.ndarray): Quality scores from generated models
        ref_distribution (np.ndarray): Quality scores from reference models
    Returns:
        dict: Dictionary of distribution metrics
    """

    # KL divergence between distributions using KDE
    x_eval = np.linspace(0, 1, 1000)
    # Estimate densities using KDE
    kde_gen = gaussian_kde(gen_distribution, bw_method='silverman')
    kde_ref = gaussian_kde(ref_distribution, bw_method='silverman')
    # Evaluate densities at the points
    p_gen = kde_gen(x_eval)
    p_ref = kde_ref(x_eval)
    # Add smoothing to avoid division by zero
    p_gen = np.maximum(p_gen, 1e-10)
    p_ref = np.maximum(p_ref, 1e-10)
    # Calculate KL divergence
    kl_div_kde = np.sum(p_ref * np.log(p_ref / p_gen)) * \
        (x_eval[1] - x_eval[0])
    
    # Jensen-Shannon distance  
    all_samples = np.concatenate([gen_distribution, ref_distribution])
    bin_edges = np.histogram_bin_edges(
        all_samples, bins='auto', range=(0, 1))
    gen_hist, _ = np.histogram(
        gen_distribution, bins=bin_edges, density=True)
    ref_hist, _ = np.histogram(
        ref_distribution, bins=bin_edges, density=True)
    # Apply Laplace smoothing (add pseudo-count)
    alpha = 0.01  # Smoothing parameter
    gen_hist_smooth = (gen_hist + alpha) / \
        (np.sum(gen_hist) + alpha * len(gen_hist))
    ref_hist_smooth = (ref_hist + alpha) / \
        (np.sum(ref_hist) + alpha * len(ref_hist))
    js_div = jensenshannon(ref_hist_smooth, gen_hist_smooth, base=2)  
    
    # Wasserstein distance 
    w_distance = wasserstein_distance(gen_distribution, ref_distribution)

    return {
        "kl_divergence_kde": kl_div_kde,
        "jensen_shannon_distance": js_div,
        "wasserstein_distance": w_distance,
    }


def compute_kid_from_embeddings(generated_embeddings, reference_embeddings):
    """
    Compute the Kernel Inception Distance (KID) between generated and reference embeddings.
    
    Args:
        generated_embeddings (torch.Tensor): Embeddings from generated images
        reference_embeddings (torch.Tensor): Embeddings from reference images
        
    Returns:
        dict: Dictionary containing KID score and related metrics
    """
    # Move tensors to CPU and convert to numpy for computation
    gen_features = generated_embeddings.cpu().numpy()
    ref_features = reference_embeddings.cpu().numpy()
        
    # Ensure we have enough samples to compute KID
    # Use minimum of 100 samples or maximum available
    n_samples = min(min(len(gen_features), len(ref_features)), 1000)
        
    # Subsample if needed to speed up computation
    if len(gen_features) > n_samples:
        indices = np.random.choice(len(gen_features), n_samples, replace=False)
        gen_features = gen_features[indices]
    if len(ref_features) > n_samples:
        indices = np.random.choice(len(ref_features), n_samples, replace=False)
        ref_features = ref_features[indices]
        
    # Compute polynomial kernel
    def polynomial_kernel(X, Y):
        # Parameters for the polynomial kernel (commonly used with KID)
        gamma = 1.0 / X.shape[1]  # 1/dimension
        coef0 = 1.0
        degree = 3
        # Compute the kernel
        K_XY = (np.matmul(X, Y.T) * gamma + coef0) ** degree
        return K_XY
        
    # Compute KID score
    # KID = (mean of K_XX entries) + (mean of K_YY entries) - 2*(mean of K_XY entries)
    K_XX = polynomial_kernel(gen_features, gen_features)
    K_YY = polynomial_kernel(ref_features, ref_features)
    K_XY = polynomial_kernel(gen_features, ref_features)
        
    # Remove diagonal elements (self-similarity) for unbiased estimation
    n_x = K_XX.shape[0]
    n_y = K_YY.shape[0]
        
    # Sum all elements and subtract diagonal, then divide by (n*(n-1))
    m_xx = (np.sum(K_XX) - np.sum(np.diag(K_XX))) / (n_x * (n_x - 1))
    m_yy = (np.sum(K_YY) - np.sum(np.diag(K_YY))) / (n_y * (n_y - 1))
        
    # Mean of cross-similarity
    m_xy = np.mean(K_XY)
        
    # Compute KID score
    kid_score = m_xx + m_yy - 2 * m_xy
        
    return {
        "kid_score": float(kid_score),
        "n_gen_samples": len(gen_features),
        "n_ref_samples": len(ref_features)
    }