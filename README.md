# GOAL

## Input:

- data matrix [X] $\in \mathbb{R}^{m \times n}$, m features, n samples, dtype=double;

- label matrix [Y] $\in \mathbb{R}^{c \times n}$, c class, dtype=double;

- dimension of subspace [d], e.g., 50, 100,...;

- adjacency [W] $\in \mathbb{R}^{n \times n}$;

- parameters [alpha],[beta],[eta],[gamma],[mu] dtype=double;

- paramter [sig_mul] dtype=int;

- #Iteration [max_Iter];
       
## Output: 
        
- projection matrix [B_mat] $\in \mathbb{R}^{m \times d}$;

- projection matrix [A_mat] $\in \mathbb{R}^{c \times d}$;

- bias vector [h_vec] $\in \mathbb{R}^c$;

- embedding in latent space [E] $\in \mathbb{R}^{n \times d}$;
        
## Usage: 

```matlab
[B, A, h, E] = func_GOAL(X, Y, d, W, ...

       alpha, beta, eta, gamma, mu, sig_mul, max_Iter);
        
low_dimentional_feature = B' * data_matrix;
```
