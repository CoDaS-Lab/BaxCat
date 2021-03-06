import baxcat.utils.cc_general_utils as utils

import math
import scipy
import random
import pylab
import numpy

from scipy.special import i0 as bessel_0
from scipy.special import i1 as bessel_1
from scipy.misc import logsumexp
from random import randrange
from math import log

from math import sin
from math import cos
from math import atan
from math import fabs

import pdb

TWOPI = 2*math.pi
LOG2PI = log(2*math.pi)

def estimate_kappa(N, ssx, scx):
    if N == 0:
        return 10.0**-6
    elif N == 1:
        return 10*math.pi
    else:
        rbar2 = (ssx/N)**2. + (scx/N)**2.
        rbar = rbar2**.5
        kappa = rbar*(2.-rbar2)/(1.-rbar2)

    A_p = lambda k : bessel_1(k)/bessel_0(k)

    Apk = A_p(kappa)
    kappa_1 = kappa - (Apk - rbar)/(1.0-Apk**2-(1.0/kappa)*Apk)
    Apk = A_p(kappa_1)
    kappa = kappa_1 - (Apk - rbar)/(1.0-Apk**2-(1.0/kappa_1)*Apk)
    Apk = A_p(kappa)
    kappa_1 = kappa - (Apk - rbar)/(1.0-Apk**2-(1.0/kappa)*Apk)
    Apk = A_p(kappa_1)
    kappa = kappa_1 - (Apk - rbar)/(1.0-Apk**2-(1.0/kappa_1)*Apk)

    if numpy.isnan(kappa):
        return 10.0**-6
    else:
        return numpy.abs(kappa)

class cc_vonmises(object):
    """
    Von Mises data type. Currently assumes fixed concentration parameter.
    All data should be in the range [0, 2*pi]
    Does not require additional argumets (distargs=None).
    """

    cctype = 'vonmises'

    def __init__(self, N=0, sum_sin_x=0, sum_cos_x=0, a=1, b=math.pi, k=1.5, distargs=None):
        """
        Keyword arguments:
        N -- numbe of data points assigned to this category. N >= 0
        sum_sin_x -- sufficient statistic sum(sin(X))
        sum_cos_x -- sufficient statistic sum(cos(X))
        a -- prior concentration parameter on Von Mises mean, mu. a > 0
        b -- prior mean parameter on Von Mises mean. b is in [0, 2*pi]
        k -- vonmises concentration parameter, k > 0
        distargs -- not used
        """
        assert N >= 0
        assert a > 0
        assert 0 <= b and b <= 2*math.pi
        assert k > 0 

        self.N = N
        self.sum_sin_x = sum_sin_x
        self.sum_cos_x = sum_cos_x
        self.a = a
        self.b = b
        self.k = k

    def set_hypers(self, hypers):
        assert hypers['a'] > 0
        assert hypers['b'] > 0
        assert 0 <= hypers['b'] and hypers['b'] <= TWOPI 

        self.a = hypers['a']
        self.b = hypers['b']
        self.k = hypers['k']

    def insert_element(self, x):
        assert 0 <= x and x <= 2*math.pi
        self.N += 1.0
        self.sum_sin_x += sin(x)
        self.sum_cos_x += cos(x)

    def remove_element(self, x):
        assert 0 <= x and x <= 2*math.pi
        self.N -= 1.0
        self.sum_sin_x -= sin(x)
        self.sum_cos_x -= cos(x)

    def predictive_logp(self, x):
        assert 0 <= x and x <= 2*math.pi
        return self.calc_predictive_logp(x, self.N, self.sum_sin_x, self.sum_cos_x,
                                        self.a, self.b, self.k)

    def singleton_logp(self, x):
        assert 0 <= x and x <= 2*math.pi
        return self.calc_predictive_logp(x, 0, 0, 0,
                                        self.a, self.b, self.k)

    def marginal_logp(self):
        logp = self.calc_marginal_logp(self.N, self.sum_sin_x, self.sum_cos_x,
                                        self.a, self.b, self.k)
        return logp

    def predictive_draw(self):
        fn = lambda x: numpy.exp(self.predictive_logp(x))
        lower_bound = 0.0
        delta = 2*math.pi/10000
        return utils.inversion_sampling(fn, lower_bound, delta)

    @staticmethod
    def construct_hyper_grids(X,n_grid=30):
        grid_interval = TWOPI/n_grid
        grids = dict()

        ssx = numpy.sum(numpy.sin(X))
        scx = numpy.sum(numpy.cos(X))
        N = float(len(X))
        k = estimate_kappa(N, ssx, scx)

        grids['a'] = utils.log_linspace(1/N,N, n_grid)
        grids['b'] = numpy.linspace(grid_interval,TWOPI, n_grid)
        grids['k'] = numpy.linspace(k,N*k, n_grid)
    
        return grids

    @staticmethod
    def init_hypers(grids, X=None):

        hypers = dict()
        hypers['a'] = random.choice(grids['a'])
        hypers['b'] = random.choice(grids['b'])
        hypers['k'] = random.choice(grids['k'])
    
        return hypers

    @staticmethod
    def calc_predictive_logp(x, N, sum_sin_x, sum_cos_x, a, b, k):
        assert 0 <= x and x <= 2*math.pi
        assert N >= 0
        assert a > 0
        assert 0 <= b and b <= 2*math.pi
        assert k > 0 

        if x < 0 or x > math.pi*2.0:
            return 0.0

        an, bn = cc_vonmises.posterior_update_parameters(
            N, sum_sin_x, sum_cos_x, a, b, k)

        am, bm = cc_vonmises.posterior_update_parameters(
            N+1.0, sum_sin_x+sin(x), sum_cos_x+cos(x), a, b, k)

        ZN = cc_vonmises.calc_log_Z(an)
        ZM = cc_vonmises.calc_log_Z(am)
    
        lp = -LOG2PI - utils.log_bessel_0(k) + ZM - ZN

        return lp

    @staticmethod
    def calc_marginal_logp(N, sum_sin_x, sum_cos_x, a, b, k):
        assert N >= 0
        assert a > 0
        assert 0 <= b and b <= 2*math.pi
        assert k > 0 
        an, bn = cc_vonmises.posterior_update_parameters(
            N, sum_sin_x, sum_cos_x, a, b, k)

        Z0 = cc_vonmises.calc_log_Z(a)
        ZN = cc_vonmises.calc_log_Z(an)

        lp = -(float(N))*(LOG2PI + utils.log_bessel_0(k)) + ZN-Z0

        if numpy.isnan(lp) or numpy.isinf(lp):
            pdb.set_trace()

        return lp
        
    @staticmethod
    def posterior_update_parameters(N, sum_sin_x, sum_cos_x, a, b, k):
        assert N >= 0
        assert a > 0
        assert 0 <= b and b <= 2*math.pi
        assert k > 0 
        
        p_cos = k*sum_cos_x+a*math.cos(b)
        p_sin = k*sum_sin_x+a*math.sin(b)

        an = (p_cos**2.0+p_sin**2.0)**.5
        bn = -1 # not used. There may be a time...

        return an, bn

    @staticmethod 
    def calc_log_Z(a):
        assert a > 0
        return utils.log_bessel_0(a)

    @staticmethod
    def update_hypers(clusters, grids):
        a = clusters[0].a
        b = clusters[0].b
        k = clusters[0].k

        which_hypers = [0,1,2]
        random.shuffle(which_hypers)

        for hyper in which_hypers:
            if hyper == 0:
                lp_a = cc_vonmises.calc_a_conditional_logps(clusters, grids['a'],b,k)
                a_index = utils.log_pflip(lp_a)
                a = grids['a'][a_index]
            elif hyper == 1:
                lp_b = cc_vonmises.calc_b_conditional_logps(clusters, grids['b'],a,k)
                b_index = utils.log_pflip(lp_b)
                b = grids['b'][b_index]
            elif hyper == 2:
                lp_k = cc_vonmises.calc_k_conditional_logps(clusters, grids['k'],a,b)
                k_index = utils.log_pflip(lp_k)
                k = grids['k'][k_index]
            else:
                raise ValueError("invalid hyper")
        
        hypers = dict()
        hypers['a'] = a
        hypers['b'] = b
        hypers['k'] = k

        return hypers
  
    @staticmethod
    def calc_a_conditional_logps(clusters, a_grid, b, k):
        lps = []
        for a in a_grid:
            lp = cc_vonmises.calc_full_marginal_conditional(clusters, a, b, k)
            lps.append(lp)

        return lps

    @staticmethod
    def calc_b_conditional_logps(clusters, b_grid, a, k):
        lps = []
        for b in b_grid:
            lp = cc_vonmises.calc_full_marginal_conditional(clusters, a, b, k)
            lps.append(lp)

        return lps

    @staticmethod
    def calc_k_conditional_logps(clusters, k_grid, a, b):
        lps = []
        for k in k_grid:
            lp = cc_vonmises.calc_full_marginal_conditional(clusters, a, b, k)
            lps.append(lp)

        return lps

    @staticmethod
    def calc_full_marginal_conditional(clusters, a, b, k):
        assert a > 0
        assert 0 <= b and b <= 2*math.pi
        assert k > 0 
        lp = 0
        for cluster in clusters:
            N = cluster.N
            sum_sin_x = cluster.sum_sin_x
            sum_cos_x = cluster.sum_cos_x
            l = cc_vonmises.calc_marginal_logp(N, sum_sin_x, sum_cos_x, a, b, k)
            lp += l

        return lp

    @staticmethod
    def calc_full_marginal_conditional_h(clusters, hypers):
        lp = 0
        a = clusters[0].a
        b = clusters[0].b
        k = clusters[0].k
        for cluster in clusters:
            N = cluster.N
            sum_sin_x = cluster.sum_sin_x
            sum_cos_x = cluster.sum_cos_x
            l = cc_vonmises.calc_marginal_logp(N, sum_sin_x, sum_cos_x, a, b, k)
            lp += l

        return lp

    @staticmethod
    def plot_dist(X, clusters, distargs=None):
        colors = ["red", "blue", "green", "yellow", "orange", "purple", "brown", "black"]
        x_min = 0
        x_max = 2*math.pi
        Y = numpy.linspace(x_min, x_max, 200)
        K = len(clusters)
        pdf = numpy.zeros((K,200))
        denom = log(float(len(X)))

        a = clusters[0].a
        b = clusters[0].b
        vmk = clusters[0].k

        nbins = min([len(X)/5, 50])

        pylab.hist(X, nbins, normed=True, color="black", alpha=.5, edgecolor="none")

        W = [log(clusters[k].N) - denom for k in range(K)]

        assert math.fabs(sum(numpy.exp(W)) -1.0) < 10.0**(-10.0)

        for k in range(K):
            w = W[k]
            N = clusters[k].N
            sum_sin_x = clusters[k].sum_sin_x
            sum_cos_x = clusters[k].sum_cos_x
            for n in range(200):
                y = Y[n]
                pdf[k, n] = numpy.exp(w + cc_vonmises.calc_predictive_logp(y, N, sum_sin_x,
                            sum_cos_x, a, b, vmk))

            if k >= 8:
                color = "white"
                alpha=.3
            else:
                color = colors[k]
                alpha=.7
            pylab.plot(Y, pdf[k,:],color=color, linewidth=5, alpha=alpha)

        pylab.plot(Y, numpy.sum(pdf,axis=0), color='black', linewidth=3)
        pylab.title('vonmises')
