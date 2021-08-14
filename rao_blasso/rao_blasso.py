import numpy as np
from numpy import linalg as np_linalg
from scipy import linalg as sp_linalg
from scipy import stats as sp_stats
from tqdm import tqdm

class RBLasso():

    def __init__(self, alpha=1.0, rao_s2=False, keep_history=False, init_ols=True):
        """Parameters:
        * alpha: the lasso (a.k.a. "lambda")  regularization constant. 
        * rao_s2: if True, use a Rao-Blackwellized estimate for the variance term.
        * keep_history: if True, keep the value of each parameter during every iteration. 
        * init_ols: if True, initialize inference with OLS solution (not possible if p > n). 
        """
        self.__alpha = alpha
        self.__rao_s2 = rao_s2
        self.__keep_history = keep_history
        self.__iterations = 0
        self.__init_ols = init_ols
        self.__beta = None
        self.__sigma = None
        self.__tau = None
        self.__X_mean = None
        self.__X_norm = None
        self.__y_mean = None

    def fit(self, X, y, num_iter=1000):
        """Fits the model, running for num_iter iterations."""

        ## compute norms and means
        self.__X_mean = np.mean(X, axis=0)
        self.__X_norm = np_linalg.norm(X, axis=0)
        self.__y_mean = np.mean(y)

        ## center X and y
        X = (X - self.__X_mean) / self.__X_norm
        y = y - self.__y_mean

        ## compute XtX and Xty
        XtX = np.dot(X.T, X)
        Xty = np.dot(X.T, y)

        ## if doing rao-blackwellized sigma, compute yTy
        if self.__rao_s2:
            yTy = np.sum(y**2)

        ## initialize beta, tau and sigma arrays, depending on keep_history flag. 
        arr_size = num_iter if self.__keep_history else 2

        ## initialize beta 
        self.__beta = np.zeros((arr_size, X.shape[1]))   
        
        if self.__init_ols and X.shape[0] > X.shape[1]:

            ## OLS estimate for beta, if possible
            chol_XtX = sp_linalg.cho_factor(XtX)
            self.__beta[0] = sp_linalg.cho_solve(chol_XtX, Xty)
        else:
            ## otherwise, random uniform
            self.__beta[0] = 2*np.random.rand(X.shape[1]) - 1.0

        ## initialize sigma with RSS of beta. 
        self.__sigma = np.zeros(arr_size)
        resid = y - np.dot(X, self.__beta[0])
        self.__sigma[0] = resid.var()
            
        ## initialize taus
        self.__tau = np.zeros((arr_size, X.shape[1]))

        ## iterate
        for cur_iter in tqdm(range(1, num_iter)):

            prev_pos = (cur_iter - 1) if self.__keep_history else (cur_iter - 1) % 2
            next_pos = cur_iter if self.__keep_history else cur_iter % 2

            ## update taus
            tau_loc = 0.5*(np.log(self.__alpha) + np.log(self.__sigma[prev_pos])) - np.log(np.abs(self.__beta[prev_pos]))
            tau_scale = self.__alpha
            self.__tau[next_pos] = sp_stats.invgauss.rvs(mu=np.exp(tau_loc) / tau_scale, scale=tau_scale)

            ## update beta
            beta_A = XtX + np.diag(self.__tau[next_pos])
            beta_A_chol = sp_linalg.cho_factor(beta_A)
            beta_mu = sp_linalg.cho_solve(beta_A_chol, Xty)
            beta_cov = sp_linalg.cho_solve(beta_A_chol, np.diag(np.repeat(self.__sigma[prev_pos], X.shape[1])))
            self.__beta[next_pos] = sp_stats.multivariate_normal.rvs(mean=beta_mu, cov=beta_cov, size=1)

            ## update sigma
            sigma_shape = ((X.shape[0] - 1) + X.shape[1]) / 2.0
            sigma_scale = None
            
            ## use rao-blackwellized estimate? 
            if self.__rao_s2:

                ## if so, use yTy
                sigma_scale = (yTy - np.dot(beta_mu.T, beta_A).dot(beta_mu)) / 2.0
            else:

                ## otherwise, compute RSS
                resid = y - np.dot(X, self.__beta[next_pos])
                sigma_scale = (np.sum(resid**2) + np.sum(self.__beta[next_pos]**2 * self.__tau[next_pos])) / 2.0
                
            self.__sigma[next_pos] = sp_stats.invgamma.rvs(a=sigma_shape, scale=sigma_scale)
            self.__iterations += 1

        return self

    def predict(self, X):
        """Computes prediction from fitted model.
        """

        if self.__iterations < 1:
            return None
        
        ## X must be centered
        X = (X - self.__X_mean) / self.__X_norm
        return np.dot(X, self.__beta[-1]) + self.__y_mean

    def get_params(self, full_history=False):
        """Returns the parameter set."""

        if (self.__iterations < 1):
            return None

        out = {
            "alpha": self.__alpha,
            "rao_s2": self.__rao_s2,
            "iterations": self.__iterations,
            "beta": None,
            "sigma": None,
            "tau": None
        }
        
        if (not self.__keep_history):
            last_pos = self.__iterations % 2            
            out["beta"] = self.__beta[last_pos]
            out["sigma"] = self.__sigma[last_pos]
            out["tau"] = self.__tau[last_pos]
            return out

        if (full_history):
            out["beta"] = self.__beta
            out["sigma"] = self.__sigma
            out["tau"] = self.__tau
            return out

        out["beta"] = self.__beta[-1]
        out["sigma"] = self.__sigma[-1]
        out["tau"] = self.__tau[-1]
        return out

