import baxcat.utils.cc_general_utils as utils

from math import log
from scipy.special import betaln

import random
import numpy
import pylab
import math

class cc_binomial(object):
    """
    Binomial data type with Beta prior on binomial parameter, theta.
    Does not require additional argumets (distargs=None).
    All X values should be 1 or 0
    """

    cctype = 'binomial'

    def __init__(self, N=0, k=0, alpha=1, beta=1, distargs=None):
        """
        Optional arguments:
        -- N: number of datapoints
        -- k: number of hits (1)
        -- alpha: beta hyperparameter
        -- beta: beta hyperparameter
        -- distargs: not used
        """
        self.N = N
        self.k = k
        self.alpha = alpha
        self.beta = beta

    def set_hypers(self, hypers):
        assert hypers['alpha'] > 0
        assert hypers['beta'] > 0
        self.alpha = hypers['alpha']
        self.beta = hypers['beta']

    def insert_element(self, x):
        assert x == 1.0 or x == 0.0
        self.N += 1
        self.k += x

    def remove_element(self, x):
        assert x == 1.0 or x == 0.0
        self.N -= 1
        self.k -= x

    def predictive_logp(self, x):
        assert x == 1.0 or x == 0.0
        return self.calc_predictive_logp(x, self.N, self.k, self.alpha, self.beta)

    def singleton_logp(self, x):
        assert x == 1.0 or x == 0.0
        return self.calc_predictive_logp(x, 0, 0, self.alpha, self.beta)

    def marginal_logp(self):
        return self.calc_marginal_logp(self.N, self.k, self.alpha, self.beta)

    def predictive_draw(self):
        if random.random() < self.alpha/(self.alpha+self.beta):
            return 1.0
        else:
            return 0.0

    @staticmethod
    def construct_hyper_grids(X,n_grid=30):
        grids = dict()
        grids['alpha'] = utils.log_linspace(1.0/float(len(X)), float(len(X)), n_grid)
        grids['beta'] = utils.log_linspace(1.0/float(len(X)), float(len(X)), n_grid)
        return grids

    @staticmethod
    def init_hypers(grids, X=None):
        hypers = dict()
        hypers['alpha'] = random.choice(grids['alpha'])
        hypers['beta'] = random.choice(grids['beta'])
        return hypers

    @staticmethod
    def calc_predictive_logp(x, N, k, alpha, beta):
        assert x == 1.0 or x == 0.0
        log_denom = log( N+alpha+beta )
        if x == 1.0:
            return log(k+alpha)-log_denom
        else:
            return log(N-k+beta)-log_denom

    @staticmethod
    def calc_marginal_logp(N, k, alpha, beta):
        lnck = utils.log_nchoosek(N, k)
        numer = betaln(k+alpha, N-k+beta)
        denom = betaln(alpha, beta)
        return lnck + numer - denom

    @staticmethod
    def update_hypers(clusters, grids):
        alpha = clusters[0].alpha
        beta = clusters[0].beta

        which_hypers = [0,1]
        random.shuffle(which_hypers)

        for hyper in which_hypers:
            if hyper == 0:
                lp_alpha = cc_binomial.calc_alpha_conditional_logps(clusters, grids['alpha'], beta)
                alpha_index = utils.log_pflip(lp_alpha)
                alpha = grids['alpha'][alpha_index]
            elif hyper == 1:
                lp_beta = cc_binomial.calc_beta_conditional_logps(clusters, grids['beta'], alpha)
                beta_index = utils.log_pflip(lp_beta)
                beta = grids['beta'][beta_index]
            else:
                raise ValueError("invalid hyper")

        hypers = dict()
        hypers['alpha'] = alpha
        hypers['beta'] = beta

        return hypers

    @staticmethod
    def calc_alpha_conditional_logps(clusters, alpha_grid, beta):
        lps = []
        for alpha in alpha_grid:
            lp = 0
            for cluster in clusters:
                N = cluster.N
                k = cluster.k
                lp += cc_binomial.calc_marginal_logp(N, k, alpha, beta)
            lps.append(lp)

        return lps

    @staticmethod
    def calc_beta_conditional_logps(clusters, beta_grid, alpha):
        lps = []
        for beta in beta_grid:
            lp = 0
            for cluster in clusters:
                N = cluster.N
                k = cluster.k
                lp += cc_binomial.calc_marginal_logp(N, k, alpha, beta)
            lps.append(lp)

        return lps

    @staticmethod
    def plot_dist(X, clusters, distargs=None):
        X_hist = numpy.histogram(X,bins=2)[0]
        X_hist = X_hist/float(len(X))
        Y = [0, 1]
        K = len(clusters)
        pdf = numpy.zeros((K,2))
        denom = log(float(len(X)))

        a = clusters[0].alpha
        b = clusters[0].beta

        pylab.bar(Y, X_hist, color="black", alpha=1, edgecolor="none")

        W = [log(clusters[k].N) - denom for k in range(K)]

        if math.fabs(sum(numpy.exp(W)) -1.0) > 10.0**(-10.0):
            pdb.set_trace()

        for k in range(K):
            w = W[k]
            N = clusters[k].N
            kk = clusters[k].k
            for n in range(2):
                y = float(Y[n])
                pdf[k, n] = numpy.exp(w + cc_binomial.calc_predictive_logp(y, N, kk, a, b))

            pylab.bar(Y, pdf[k,:],color="white", edgecolor="none", alpha=.5)

        pylab.bar(Y, numpy.sum(pdf,axis=0), color='none', edgecolor="red", linewidth=3)
        pylab.xlim([-.1,1.9])
        pylab.ylim([0,1.0])
        pylab.title('binomial')
