import math
from math import log
import baxcat.utils.cc_general_utils as utils
import baxcat.utils.cc_sample_utils as su
from scipy.special import gammaln
from random import randrange
import random
import numpy
import scipy

from scipy.misc import logsumexp

import pylab


class cc_beta_uc(object):
    """
    Beta dsitiburion reparameterized in terms of strength, s, and balance, b:
        θ ~ Beta(s*b, s*(1-b)),
        s ~ exp(λ),
        b ~ Beta(α,β).
    Does not require additional argumets (distargs=None).
    """

    cctype = 'beta_uc'

    def __init__(self, N=0, sum_log_x=0, sum_minus_log_x=0, strength=2.0, balance=.5, λ=1, α=.5, β=.5, distargs=None):
        """
        Optional arguments:
        -- N: number of data points
        -- sum_log_x: suffstat, sum(log(X))
        -- sum_minus_log_x: suffstat, sum(log(1-X))
        -- strength: higher strength -> lower variance
        -- balance: analogous to mean
        -- λ: hyperparameter, exponential distribution parameter for prior on strength
        -- α: hyperparameter, beta distribution parameter for balance
        -- β: hyperparameter, beta distribution parameter for balance
        -- distargs: not used
        """
        
        self.N = N
        self.sum_log_x = sum_log_x
        self.sum_minus_log_x = sum_minus_log_x

        self.strength = strength
        self.balance = balance

        self.λ = λ
        self.α = α
        self.β = β

    def insert_element(self, x):
        assert x > 0 and x < 1
        self.N += 1.0
        self.sum_log_x += log(x)
        self.sum_minus_log_x += log(1.0-x)

    def remove_element(self, x):
        assert x > 0 and x < 1
        self.N -= 1.0
        if self.N <= 0:
            self.sum_log_x = 0
            self.sum_minus_log_x = 0
        else:
            self.sum_log_x -= log(x)
            self.sum_minus_log_x -= log(1.0-x)

    def set_component_params_from_prior(self):
        self.strength, self.balance = cc_beta_uc.draw_params(self.λ, self.α, self.β)

    def update_component_parameters(self):
        # resample mu
        n_samples = 25
    
        log_pdf_lambda_str = lambda strength : cc_beta_uc.calc_logp(self.N, self.sum_log_x, self.sum_minus_log_x, strength, self.balance)+cc_beta_uc.calc_log_prior(strength, self.balance, self.λ, self.α, self.β)
        self.strength = su.mh_sample(self.strength, log_pdf_lambda_str, .5, [0.0,float('Inf')], burn=n_samples)

        log_pdf_lambda_bal = lambda balance : cc_beta_uc.calc_logp(self.N, self.sum_log_x, self.sum_minus_log_x, self.strength, balance)+cc_beta_uc.calc_log_prior(self.strength, balance, self.λ, self.α, self.β)
        self.balance = su.mh_sample(self.balance, log_pdf_lambda_bal, .25, [0,1], burn=n_samples)

    def set_params(self, params):
        assert params['strength'] > 0
        assert params['balance'] >= 0 and params['balance'] <= 1
        self.strength = params['strength']
        self.balance = params['balance']

    def predictive_logp(self, x):
        return cc_beta_uc.calc_singleton_logp(x, self.strength, self.balance)

    def predictive_draw(self):
        α = self.strength*self.balance
        β = self.strength*(1.0-self.balance)
        return numpy.random.beta(α, β)

    def marginal_logp(self):
        lp = cc_beta_uc.calc_logp(self.N, self.sum_log_x, self.sum_minus_log_x, self.strength, self.balance)
        lp += cc_beta_uc.calc_log_prior(self.strength, self.balance, self.λ, self.α, self.β)
        return lp

    def set_hypers(self, hypers):
        self.λ = hypers['λ']
        self.α = hypers['α']
        self.β = hypers['β']

    @staticmethod
    def singleton_logp(x, hypers):
        λ = hypers['λ']
        α = hypers['α']
        β = hypers['β']
        strength, balance = cc_beta_uc.draw_beta_params(λ, α, β)
        logp = cc_beta_uc.calc_singleton_logp(x, strength, balance)
        params = dict( strength=strength, balance=balance)

        return logp, params

    @staticmethod
    def calc_singleton_logp(x, strength, balance):
        assert( strength > 0 and balance > 0 and balance < 1)
            
        α = strength*balance
        β = strength*(1.0-balance)
        lp = scipy.stats.beta.logpdf(x, α, β)

        assert( not numpy.isnan(lp) )
        return lp

    @staticmethod
    def calc_log_prior(strength, balance, λ, α, β):
        assert( strength > 0 and balance > 0 and balance < 1)

        lp = 0
        lp += scipy.stats.expon.logpdf(strength, scale=λ)
        lp += scipy.stats.beta.logpdf(balance, α, β)
        return lp

    @staticmethod
    def calc_logp(N, sum_log_x, sum_minus_log_x, strength, balance):
        assert( strength > 0 and balance > 0 and balance < 1)

        α = strength*balance
        β = strength*(1.0-balance)
        lp = 0
        lp -= N*scipy.special.betaln(α, β)
        lp += (α-1.0)*sum_log_x
        lp += (β-1.0)*sum_minus_log_x
        
        assert( not numpy.isnan(lp) )
        return lpassert( not numpy.isnan(lp) )

    @staticmethod
    def draw_beta_params(λ, α, β):
        strength = numpy.random.exponential(scale=λ)
        balance = numpy.random.beta(α, β)

        assert( strength > 0 and balance > 0 and balance < 1)

        return strength, balance

    @staticmethod
    def construct_hyper_grids(X, n_grid=30):
        N = float(len(X))
        Sx = numpy.sum(X)
        Mx = numpy.sum(1-X)
        grids = {
            'λ' : utils.log_linspace(1/N, N, n_grid),
            'α' : utils.log_linspace(Sx/N, Sx, n_grid),
            'β' : utils.log_linspace(Mx/N, Mx, n_grid),
        }
        return grids

    @staticmethod
    def init_hypers(grids, X=None):
        hypers = {
            'λ' : random.choice(grids['λ']),
            'α' : random.choice(grids['α']),
            'β' : random.choice(grids['β']),
        }
        return hypers

    @staticmethod
    def calc_prior_conditionals(clusters, λ, α, β):
        lp = 0
        for cluster in clusters:
            strength = cluster.strength
            balance = cluster.balance
            lp += cc_beta_uc.calc_log_prior(strength, balance, λ, α, β)
        return lp

    @staticmethod
    def calc_λ_conditional_logps(clusters, λ_grid, α, β):
        lps = []
        for λ in λ_grid:
            lps.append( cc_beta_uc.calc_prior_conditionals(clusters, λ, α, β) )
        return lps

    @staticmethod
    def calc_α_conditional_logps(clusters, α_grid, λ, β):
        lps = []
        for α in α_grid:
            lps.append( cc_beta_uc.calc_prior_conditionals(clusters, λ, α, β) )
        return lps

    @staticmethod
    def calc_β_conditional_logps(clusters, β_grid, λ, α):
        lps = []
        for β in β_grid:
            lps.append( cc_beta_uc.calc_prior_conditionals(clusters, λ, α, β) )
        return lps


    @staticmethod
    def update_hypers(clusters, grids):
        # resample hypers
        λ = clusters[0].λ
        α = clusters[0].α
        β = clusters[0].β

        which_hypers = [0,1,2]
        random.shuffle(which_hypers)

        for hyper in which_hypers:
            if hyper == 0:
                lp_λ = cc_beta_uc.calc_λ_conditional_logps(clusters, grids['λ'], α, β)
                λ_index = utils.log_pflip(lp_λ)
                λ = grids['λ'][λ_index]
            elif hyper == 1:
                lp_α = cc_beta_uc.calc_α_conditional_logps(clusters, grids['α'], λ, β)
                α_index = utils.log_pflip(lp_α)
                α = grids['α'][α_index]
            elif hyper == 2:
                lp_β = cc_beta_uc.calc_β_conditional_logps(clusters, grids['β'], λ, α)
                β_index = utils.log_pflip(lp_β)
                β = grids['β'][β_index]
            else:
                raise ValueError("invalild hyper")
        
        hypers = dict()
        hypers['λ'] = λ
        hypers['α'] = α
        hypers['β'] = β
        
        return hypers
  

    @staticmethod
    def plot_dist(X, clusters, distargs=None):
        colors = ["red", "blue", "green", "yellow", "orange", "purple", "brown", "black"]
        N = 100
        Y = numpy.linspace(0.01, .99, N)
        K = len(clusters)
        pdf = numpy.zeros((K,N))
        denom = log(float(len(X)))

        nbins = min([len(X)/5, 50])

        pylab.hist(X, nbins, normed=True, color="black", alpha=.5, edgecolor="none")

        W = [log(clusters[k].N) - denom for k in range(K)]

        for k in range(K):
            w = W[k]
            strength = clusters[k].strength
            balance = clusters[k].balance
            for n in range(N):
                y = Y[n]
                pdf[k, n] = numpy.exp(w + cc_beta_uc.calc_singleton_logp(y, strength, balance))

            if k >= 8:
                color = "white"
                alpha=.3
            else:
                color = colors[k]
                alpha=.7
            pylab.plot(Y, pdf[k,:],color=color, linewidth=5, alpha=alpha)

        pylab.plot(Y, numpy.sum(pdf,axis=0), color='black', linewidth=3)
        pylab.title('beta (uncollapsed)')
