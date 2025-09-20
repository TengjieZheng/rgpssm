## Improvements for the RGPSSM implementation

1. **Custom Kernel Implementation**  
   - Replaced the `gpytorch` kernel with a self-implemented kernel function, leading to a **~40% runtime speedup**. 

2. **Reordered State and Inducing Points in Covariance Matrix**  
   - Changed the ordering of state `x` and inducing points `u` in the joint covariance matrix `S`.  
   - In the new design, `Suu` is placed in the **top-left corner** of the joint covariance.  
   - This enables **direct reuse of the Cholesky factor of `Suu`** during factor updates, reducing the computational complexity of the Cholesky-based prediction step from **cubic to quadratic**.  
   - **Code locations:**  
     - Original: `model/rgpssm/rgpssm.py`  
     - Updated: `model/rgpssm/rgpssm_u.py`

3. **Interface Updates**  
   - The API definition differs slightly from the first version of RGPSSM.  
   - Please refer to the provided **experimental code** for correct usage.

---