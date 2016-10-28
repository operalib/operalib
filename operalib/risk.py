"""
:mod:`operalib.risk` implements risk model and their gradients.
"""
# Authors: Romain Brault <romain.brault@telecom-paristech.fr> with help from
#         the scikit-learn community.
#         Maxime Sangnier <maxime.sangnier@gmail.com>
# License: MIT

from numpy.linalg import norm
from numpy import inner
from numpy import kron,bmat
from scipy.linalg import block_diag


class KernelRidgeRisk(object):
    """Define Kernel ridge risk and its gradient."""

    def __init__(self, lbda):
        """Initialize Empirical kernel ridge risk.

        Parameters
        ----------
        lbda : {float}
            Small positive values of lbda improve the conditioning of the
            problem and reduce the variance of the estimates.  Lbda corresponds
            to ``(2*C)^-1`` in other linear models such as LogisticRegression
            or LinearSVC.
            
        """
        self.lbda = lbda

    def __call__(self, coefs, ground_truth, Gram):
        """Compute the Empirical OVK ridge risk.

        Parameters
        ----------
        coefs : {vector-like}, shape = [n_samples1 * n_targets]
            Coefficient to optimise

        ground_truth : {vector-like}
            Targets samples

        Gram : {LinearOperator}
            Gram matrix acting on the coefs

        Returns
        -------
        float : Empirical OVK ridge risk
        """
        pred = Gram * coefs
        res = pred - ground_truth
        np = ground_truth.size
        reg = inner(coefs, pred)
        return norm(res) ** 2 / (2 * np) + self.lbda * reg / (2 * np)

    def functional_grad(self, coefs, ground_truth, Gram):
        """Compute the gradient of the Empirical OVK ridge risk.

        Parameters
        ----------
        coefs : {vector-like}, shape = [n_samples1 * n_targets]
            Coefficient to optimise

        ground_truth : {vector-like}
            Targets samples

        Gram : {LinearOperator}
            Gram matrix acting on the coefs

        Returns
        -------
        {vector-like} : gradient of the Empirical OVK ridge risk
        """
        pred = Gram * coefs
        res = pred - ground_truth
        np = ground_truth.size
        return Gram * res / np + self.lbda * pred / np

    def functional_grad_val(self, coefs, ground_truth, Gram):
        """Compute the gradient and value of the Empirical OVK ridge risk.

        Parameters
        ----------
        coefs : {vector-like}, shape = [n_samples1 * n_targets]
            Coefficient to optimise

        ground_truth : {vector-like}
            Targets samples

        Gram : {LinearOperator}
            Gram matrix acting on the coefs

        Returns
        -------
        Tuple{float, vector-like} : Empirical OVK ridge risk and its gradient
        returned as a tuple.
        """
        pred = Gram * coefs
        res = pred - ground_truth
        np = ground_truth.size
        reg = inner(coefs, pred)
        return (norm(res) ** 2 / (2 * np) + self.lbda * reg / (2 * np), Gram *
                res / np + self.lbda * pred / np)

class WeakSupKernelRidgeRisk(object):
    """Define Kernel ridge risk with weak supervision formulation and its gradient."""
    
    def __init__(self):
        """Initialize Empirical kernel ridge risk.

        Parameters
        ----------
        lbda0 : {float}
            Positive value controlling the regularization strength in the RKHS
        lbda1 : {float}
            Positive value controlling the weight of the fully labelled data
        lbda2 : {float}
            Positive value controlling the weight of the weakly labelled data
        nfeatweak : {int}
            Number of features non zero for the weakly labelled data (must be 
        provided in the first columns of the training data)
        ndataweak : {int}
            Number of examples used as the weakly labelled data (must be 
        provided in the last rows of the training data)
              
        """
        self.lbda0 = lbda0
        self.lbda1 = lbda1
        self.lbda2 = lbda2
        self.nfeatweak = nfeatweak
        self.ndataweak = ndataweak

                
    def __call__(self, coefs, ground_truth, Gram):
        """Compute the Empirical OVK ridge weak risk.

        Parameters
        ----------
        coefs : {vector-like}, shape = [n_samples1 * n_targets]
            Coefficient to optimise

        ground_truth : {vector-like}
            Targets samples

        Gram : {LinearOperator}
            Gram matrix acting on the coefs

        Returns
        -------
        float : Empirical OVK ridge weak risk
        """
        pred = Gram * coefs
        res = pred - ground_truth
        np = ground_truth.size
        reg = inner(coefs, pred)
        return norm(res) ** 2 / (2 * np) + self.lbda * reg / (2 * np)
        
    def functional_grad(self, coefs, ground_truth, Gram):
        """Compute the gradient of the Empirical OVK ridge weak risk.

        Parameters
        ----------
        coefs : {vector-like}, shape = [n_samples1 * n_targets]
            Coefficient to optimise

        ground_truth : {vector-like}
            Targets samples

        Gram : {LinearOperator}
            Gram matrix acting on the coefs

        Returns
        -------
        {vector-like} : gradient of the Empirical OVK ridge weak risk
        """
        n_training = Gram.size
        y_tilde_1 = ground_truth[:self.nfeatweak,:]
        y_tilde_2 = ground_truth[self.nfeatweak:,:]
        d_y_tilde_1 = bmat([[y_tilde_1],[y_tilde_1]])
        d_y_tilde_2 = bmat([[y_tilde_2],[y_tilde_2]])
        
        
        pred = Gram * coefs
        
        # Extract submatrices corresponding to fully labeled and weak parts
        K1 = Gram[:n_training-self.ndataweak,:n_training-self.ndataweak]
        K2 = Gram[:n_training-self.ndataweak,n_training-self.ndataweak:]
        K3 = K2.T
        K4 = Gram[n_training-self.ndataweak:,n_training-self.ndataweak:]
        
        # First auxiliary gram matrix :
        U = bmat([[K1.T * K1 , K1 * K3],[K3 * K1 , K3 * K2]]) 
        Uc = U * coefs
        
        # Second auxiliary gram matrix :
        V = block_diag(K1,K3)
        Vy = V * d_y_tilde_1
        
        # Auxiliray gram matrices for the weak part :
        A = np.bmat([[np.eye(nfeatweak) , np.zeros([nfeatweak,coefs.shape[1]])],
        [np.zeros([coefs.shape[1],nfeatweak]), np.zeros([coefs.shape[1],coefs.shape[1]])]])
        Kprime1 = kron(K2 * K3,A)
        Kprime2 = kron(K3 * K4,A)
        Kprime3 = kron(K4 * K3,A)
        Kprime4 = kron(K4 * K4,A)
        K3kronA = kron(K3,A)
        K4kronA = kron(K4,A)
        
        # Third Auxiliary matrix
        W = bmat([[Kprime1 , Kprime2 ] , [Kprime3 , Kprime4]])
        Wc = W * coefs
        
        # Fourth Auxiliary matrix 
        Z = block_diag(K3kronA,K4kronA)
        Zy = Z * d_y_tilde_2
        
        
        return 2*(self.lbda0 * pred + self.lbda1 * (Uc - Vy) + self.lbda2 * (Wc - Zy))
        
        
        #res = pred - ground_truth
        #np = ground_truth.size
        #return Gram * res / np + self.lbda * pred / np
        
    def functional_grad_val(self, coefs, ground_truth, Gram):
        """Compute the gradient and value of the Empirical OVK ridge risk.

        Parameters
        ----------
        coefs : {vector-like}, shape = [n_samples1 * n_targets]
            Coefficient to optimise

        ground_truth : {vector-like}
            Targets samples

        Gram : {LinearOperator}
            Gram matrix acting on the coefs

        Returns
        -------
        Tuple{float, vector-like} : Empirical OVK ridge risk and its gradient
        returned as a tuple.
        """
        n_training = Gram.size
        y_tilde_1 = ground_truth[:self.nfeatweak,:]
        y_tilde_2 = ground_truth[self.nfeatweak:,:]
        d_y_tilde_1 = bmat([[y_tilde_1],[y_tilde_1]])
        d_y_tilde_2 = bmat([[y_tilde_2],[y_tilde_2]])
        
        
        pred = Gram * coefs
        
        # Extract submatrices corresponding to fully labeled and weak parts
        K1 = Gram[:n_training-self.ndataweak,:n_training-self.ndataweak]
        K2 = Gram[:n_training-self.ndataweak,n_training-self.ndataweak:]
        K3 = K2.T
        K4 = Gram[n_training-self.ndataweak:,n_training-self.ndataweak:]
        
        # First auxiliary gram matrix :
        U = bmat([[K1.T * K1 , K1 * K3],[K3 * K1 , K3 * K2]]) 
        Uc = U * coefs
        
        # Second auxiliary gram matrix :
        V = block_diag(K1,K3)
        Vy = V * d_y_tilde_1
        
        # Auxiliray gram matrices for the weak part :
        A = np.bmat([[np.eye(nfeatweak) , np.zeros([nfeatweak,coefs.shape[1]])],
        [np.zeros([coefs.shape[1],nfeatweak]), np.zeros([coefs.shape[1],coefs.shape[1]])]])
        Kprime1 = kron(K2 * K3,A)
        Kprime2 = kron(K3 * K4,A)
        Kprime3 = kron(K4 * K3,A)
        Kprime4 = kron(K4 * K4,A)
        K3kronA = kron(K3,A)
        K4kronA = kron(K4,A)
        
        # Third Auxiliary matrix
        W = bmat([[Kprime1 , Kprime2 ] , [Kprime3 , Kprime4]])
        Wc = W * coefs
        
        # Fourth Auxiliary matrix 
        Z = block_diag(K3kronA,K4kronA)
        Zy = Z * d_y_tilde_2
        
        val = inner(coefs , self.lbda0 * pred + self.lbda1 * (Uc - 2*Vy) + self.lbda2 * (Wc - 2*Zy))
        
        return val , 2*(self.lbda0 * pred + self.lbda1 * (Uc - Vy) + self.lbda2 * (Wc - Zy)) 
        
        
        #pred = Gram * coefs
        #res = pred - ground_truth
        #np = ground_truth.size
        #reg = inner(coefs, pred)
        #return (norm(res) ** 2 / (2 * np) + self.lbda * reg / (2 * np), Gram *
        #        res / np + self.lbda * pred / np)        
