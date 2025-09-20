## Implementation for RGPSSM-H and Relevant Experiment Code

In this implementation, we also improved the original RGPSSM method. The main improvements are summarized below.

### Improvements for the RGPSSM Implementation

1. **Custom Kernel Implementation**  
   - Replaced the `gpytorch` kernel with a self-implemented kernel function.  
   - Removed redundant overhead, achieving a **~40% runtime speedup**.

2. **Reordered State and Inducing Points in the Covariance Matrix**  
   - Adjusted the ordering of state `x` and inducing points `u` in the joint covariance matrix `S`.  
   - In the new design, `Suu` is placed in the **top-left corner** of the joint covariance.  
   - This allows **direct reuse of the Cholesky factor of `Suu`** during factor updates, reducing the computational complexity of the Cholesky-based prediction step from **cubic to quadratic**.  
   - **Code locations:**  
     - Original: `model/rgpssm/rgpssm.py`  
     - Updated: `model/rgpssm/rgpssm_u.py`

3. **Interface Updates**  
   - The API definition differs slightly from the first version of RGPSSM.  
   - Please refer to the provided **experiment code** for correct usage.
