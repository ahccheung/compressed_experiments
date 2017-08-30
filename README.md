# Learning gene modules

This project contains the code for simulating gene expression profiles and learning gene module networks from the generated data.

## Generating simulated data

We simulate gene expression profiles in the following manner:

 1. First generate N pre-stimulus profiles x by a Poisson random variable, with
*s* samples per unique experiment. Each experiment corresponds to a random vector of expected values (lambda).
 2. Generate random tensor of rank L and order d by using a shallow decomposition.
 3. Simulate post-stimulus profiles y according to the random tensor and x. Each y is a random poisson variable: Y~ Poisson(M.dot(A(x)) + b), where b is a bias vector and M is the weight matrix for the module A(x).

To generated simulated data, run:
`python generate_data.py N J G d k L s file_label`

Arguments:
 1. N = Number of samples
 2. J = Dimension of pre-stimulus profiles
 3. G = Dimension of post-stimulus profiles
 4. d = order of tensor
 5. L = rank of tensor
 6. s = number of samples per experiment
 7. file_label = prefix for output files.

Output files:
 1. Training and testing sets: file_label + 'trainx', file_label + 'trainy', file_label + 'testx', file_label + 'testy'
 2.
