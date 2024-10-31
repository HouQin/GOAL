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

## Citation
```bib
@ARTICLE{GOAL2024lu,
       author={Lu, Haoquan and Lai, Zhihui and Zhang, Junhong and Yu, Zhuozhen and Wen, Jiajun},
       journal={ IEEE Transactions on Artificial Intelligence },
       title={{ GOAL: Generalized Jointly Sparse Linear Discriminant Regression for Feature Extraction }},
       year={2024},
       volume={5},
       number={10},
       ISSN={2691-4581},
       pages={4959-4971},
       doi={10.1109/TAI.2024.3412862},
       url = {https://doi.ieeecomputersociety.org/10.1109/TAI.2024.3412862},
       publisher={IEEE Computer Society},
       address={Los Alamitos, CA, USA},
       month=oct
}
```
