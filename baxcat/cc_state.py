import numpy
import random
import pylab
from math import log
from math import exp
from scipy.stats import gamma
from scipy.stats import expon
from scipy.misc import logsumexp
from numpy.random import gamma as gamrnd
from numpy.random import exponential as exprnd

import baxcat.utils.cc_general_utils as utils
import baxcat.utils.cc_plot_utils as pu
import baxcat.utils.cc_sample_utils as su

from baxcat.cc_types import cc_normal_uc
from baxcat.cc_types import cc_beta_uc
from baxcat.cc_types import cc_normal_model
from baxcat.cc_types import cc_binomial_model
from baxcat.cc_types import cc_multinomial_model
from baxcat.cc_types import cc_lognormal_model
from baxcat.cc_types import cc_poisson_model
from baxcat.cc_types import cc_vonmises_model
from baxcat.cc_types import cc_vonmises_uc

from baxcat.cc_view import cc_view
from baxcat.cc_dim import cc_dim
from baxcat.cc_dim_uc import cc_dim_uc

_is_uncollapsed = {
    'normal'      : False,
    'normal_uc'   : True,
    'beta_uc'     : True,
    'binomial'    : False,
    'multinomial' : False,
    'lognormal'   : False,
    'poisson'     : False,
    'vonmises'    : False,
    'vonmises_uc' : True,
    } 

_cctype_class = {
    'normal'      : cc_normal_model.cc_normal,
    'normal_uc'   : cc_normal_uc.cc_normal_uc,
    'beta_uc'     : cc_beta_uc.cc_beta_uc,
    'binomial'    : cc_binomial_model.cc_binomial,
    'multinomial' : cc_multinomial_model.cc_multinomial,
    'lognormal'   : cc_lognormal_model.cc_lognormal,
    'poisson'     : cc_poisson_model.cc_poisson,
    'vonmises'    : cc_vonmises_model.cc_vonmises,
    'vonmises_uc' : cc_vonmises_uc.cc_vonmises_uc,
    }

_all_kernels = ['column_z','state_alpha','row_z','column_hypers','view_alphas']


class cc_state(object):
    """
    cc_state. The main crosscat object.
    properties:
    -- n_rows: (int) number of rows
    -- n_cols: (int) number of columns
    -- n_grid: (int) number of bins in hyperparameter grids
    """
         
    def __init__(self, X, cctypes, distargs, n_grid=30, Zv=None, Zrcv=None, hypers=None, seed=None):
        """
        cc_state constructor

        input arguments:
        -- X: a list of numpy data columns.
        -- cctypes: a list of strings where each entry is the data type for 
        each column.
        -- distargs: a list of distargs appropriate for each type in cctype.
        For details on distrags see the documentation for each data type.

        optional arguments:
        -- n_grid: number of bins for hyperparameter grids. Default = 30.
        -- Zv: The assignment of columns to views. If not specified, a 
        partition is generated randomly
        -- Zrcv: The assignment of rows to clusters for each view
        -- ct_kernel: which column transition kenerl to use. Default = 0 (Gibbs)
        -- seed: seed the random number generator. Default = system time.

        example:
        >>> import numpy
        >>> n_rows = 100
        >>> X = [numpy.random.normal(n_rows), numpy.random.normal(n_rows)]
        >>> State = cc_state(X, ['normal', 'normal'], [None, None])
        """

        if seed is not None:
            random.seed(seed)
            numpy.random.seed(seed)

        self.__column_transition_kernels = [
            self.__transition_columns_kernel_0,     # orignal Gibbs (no gaps)
            self.__transition_columns_kernel_1,     # Metropolis-Hastings
            self.__transition_columns_kernel_2,     # Gibbs with aux parameters
        ]
        
        self.n_rows = len(X[0])
        self.n_cols = len(X)
        self.n_grid = n_grid

        # construct the dims
        self.dims = []
        for col in range(self.n_cols):
            Y = X[col] 
            cctype = cctypes[col]
            if _is_uncollapsed[cctype]:
                dim = cc_dim_uc(Y, _cctype_class[cctype], col, n_grid=n_grid, distargs=distargs[col])
            else:
                dim = cc_dim(Y, _cctype_class[cctype], col, n_grid=n_grid, distargs=distargs[col])
            self.dims.append(dim)

        # set the hyperparameters in the dims
        if hypers is not None:
            for d in range(self.n_cols):
                self.dims[d].set_hypers(hypers[d])

        # initialize CRP alpha  
        self.alpha_grid = utils.log_linspace(1.0/self.n_cols, self.n_cols, self.n_grid)
        self.alpha = random.choice(self.alpha_grid)

        assert len(self.dims) == self.n_cols

        if Zrcv is not None:
            assert Zv is not None
            assert len(Zv) == self.n_cols
            assert len(Zrcv) == max(Zv)+1
            assert len(Zrcv[0]) == self.n_rows

        # construct the view partition
        if Zv is None:
            Zv, Nv, V = utils.crp_gen(self.n_cols, self.alpha)
        else:
            Nv = utils.bincount(Zv)
            V = len(Nv)

        # construct views
        self.views = []
        for view in range(V):
            indices = [i for i in range(self.n_cols) if Zv[i] == view]
            dims_view = []
            for index in indices:
                dims_view.append(self.dims[index])

            if Zrcv is None:
                self.views.append(cc_view(dims_view, n_grid=n_grid))
            else:
                self.views.append(cc_view(dims_view, Z=numpy.array(Zrcv[view]), n_grid=n_grid))

        self.Zv = numpy.array(Zv)
        self.Nv = Nv
        self.V = V

    @classmethod
    def from_metadata(cls, X, metadata):
        
        Zv = metadata['Zv']
        Zrcv = metadata['Zrcv']
        n_grid = metadata['n_grid']
        hypers = metadata['hypers']
        cctypes = metadata['cctypes']
        distargs = metadata['distargs']

        return cls(X, cctypes, distargs, n_grid, Zv, Zrcv, hypers)

    def get_metadata(self):
        metadata = dict()
        
        # misc data
        metadata['n_grid'] = self.n_grid

        # view data
        metadata['V'] = self.V
        metadata['Nv'] = self.Nv
        metadata['Zv'] = self.Zv

        # category data
        metadata['K'] = []
        metadata['Nk'] = []
        metadata['Zrcv'] = []

        # column data
        metadata['hypers'] = []
        metadata['cctypes'] = []
        metadata['distargs'] = []
        metadata['suffstats'] = []

        for dim in self.dims:
            metadata['hypers'].append(dim.hypers)
            metadata['distargs'].append(dim.distargs)
            metadata['cctypes'].append(dim.cctype)
            metadata['suffstats'].append(dim.get_suffstats())

        for view in self.views:
            metadata['K'].append(view.K)
            metadata['Nk'].append(view.Nk)
            metadata['Zrcv'].append(view.Z)

        return metadata
            

        
    def append_dim(self, X_f, cctype, distargs=None, ct_kernel=0, m=1):
        """
        Add a new data column to X.
        Inputs:
        -- X_f: a numpy array of data
        -- cctype: type of the data
        Keyword args:
        -- distargs: for multinomial data
        -- ct_kernel: must be 0 or 2. MH kernel cannot be used to append
        -- m: for ct_kernel=2. Number of auxiliary parameters
        """

        col = self.n_cols
        n_grid = self.n_grid

        if _is_uncollapsed[cctype]:
            dim = cc_dim_uc(X_f, _cctype_class[cctype], col, n_grid=n_grid, distargs=distargs)
        else:
            dim = cc_dim(X_f, _cctype_class[cctype], col, n_grid=n_grid, distargs=distargs)

        self.n_cols += 1

        self.dims.append(dim)
        self.Zv = numpy.append(self.Zv, -1)
        
        if ct_kernel == 2:
            self.__column_transition_kernels[ct_kernel](m=m, append=True)
        elif ct_kernel == 0:
            self.__column_transition_kernels[ct_kernel](append=True)
        else:
            raise ValueError("invalid ct_kernel (%s) for append_dim." % (str(ct_kernel)))

        self.__check_partitions()

    def transition(self, N=1, kernel_list=None, ct_kernel=0, which_rows=None, which_cols=None, do_plot=False):
        """
        Do transitions.

        optional arguments:
        -- N: number of transitions.
        -- kernel_list: which kernels to do.
        -- ct_kernel: which column transitino kernel to use {0,1,2} = {Gibbs, MH, Aux Gibbs}
        -- which_rows: list of rows to apply the transitions to
        -- which_cols: list of columns to apply the transitions to
        -- do_plot: plot the state of the sampler (real-time)
        examples:
        >>> State.transition()
        >>> State.transition(N=100)
        >>> State.transition(N=100, kernel_list=['column_z','row_z'])
        """

        kernel_dict = {
            'column_z'      : lambda : self.__transition_columns(which_cols,ct_kernel),
            'state_alpha'   : lambda : self.__transition_state_alpha(),
            'row_z'         : lambda : self.__transition_rows(which_rows),
            'column_hypers' : lambda : self.__transition_column_hypers(which_rows),
            'view_alphas'   : lambda : self.__transition_view_alphas(),
        }

        if kernel_list is None:
            kernel_list = _all_kernels

        kernel_fns = [ kernel_dict[kernel] for kernel in kernel_list ]

        if do_plot:
            pylab.ion()
            layout = pu.get_state_plot_layout(self.n_cols)
            fig = pylab.figure(num=None, figsize=(layout['plot_inches_y'], layout['plot_inches_x']), 
                dpi=75, facecolor='w', edgecolor='k',frameon=False,tight_layout=True)
            self.__plot(fig,layout)

        for i in range(N):
            # random.shuffle(kernel_fns)
            for kernel in kernel_fns:
                kernel()
            if do_plot:
                self.__plot(fig,layout)

    def set_data(self, data):
        """
        Testing. Resets the suffstats in all clusters in all dims to reflect
        the new data.
        """
        for col in range(self.n_cols):
            self.dims[col].X = data[col]
            self.dims[col].reassign(self.views[self.Zv[col]].Z)

    def dump_data(self):
        """
        Testing. Clears the suffstats in all clusters in all dims.
        """
        for dim in self.dims:
            dim.dump_data()

    def _update_prior_grids(self):
        for dim in self.dims:
            dim.update_prior_grids()

    def __transition_columns(self, which_cols=None, ct_kernel=0):
        """Transition column assignment to views"""
        column_transition_kernel = self.__column_transition_kernels[ct_kernel]
        column_transition_kernel(which_cols=None)

    def __transition_columns_kernel_0(self, which_cols=None, append=False):
        """ 
        column assignment transition kernel 0: Gibbs
        """
        if append:
            which_cols = [self.n_cols-1]

        if which_cols is None:
            which_cols = [i for i in range(self.n_cols)]

        random.shuffle(which_cols)

        for col in which_cols:

            # get starting view and check whether it is a singleton
            v_a = self.Zv[col] 

            if append:
                is_singleton = False
                pv = list(self.Nv)
            else:
                is_singleton = (self.Nv[v_a] == 1)

                pv = list(self.Nv)
                # remove value from CRP
                if is_singleton:
                    # if v_a is a singleton, do not consider move to new singleton view
                    pv[v_a] = self.alpha
                else:
                    pv[v_a] -= 1
                    # must consider singleton
                    pv.append(self.alpha)

            # take log of all values (no need to worry about denominator)
            pv = numpy.log(numpy.array(pv))

            ps = []
            dim = self.dims[col]
            for v in range(self.V):
                # Get the probability of the data column under each view's assignment
                dim.reassign(self.views[v].Z)
                p_v = dim.full_marginal_logp()+pv[v]
                ps.append(p_v)

            # If v_a is not a singleton propose one (drawn from the prior)
            if not is_singleton:
                proposal_view = cc_view([dim], n_grid=self.n_grid)
                dim.reassign(proposal_view.Z)
                p_v = dim.full_marginal_logp()+pv[-1]
                ps.append(p_v)

            # draw a new view
            v_b = utils.log_pflip(ps)

            if append:
                self.__append_new_dim_to_view(dim, v_b, proposal_view)
                continue

            # clean up
            if v_b != v_a:
                if is_singleton:
                    self.__destroy_singleton_view(dim, v_a, v_b)
                elif v_b == self.V:
                    self.__create_singleton_view(dim, v_a, proposal_view)
                else:
                    self.__move_dim_to_view(dim, v_a, v_b)
            else:
                self.dims[col].reassign(self.views[v_a].Z)

            # self.__check_partitions()

    def __transition_columns_kernel_1(self, which_cols=None):
        """column assignment transition kernel 1: Metropolis Birth-Death process"""

        if which_cols is None:
            which_cols = [i for i in range(self.n_cols)]

        random.shuffle(which_cols)

        for col in which_cols:
            
            # this dim
            dim = self.dims[col]

            # get start view, v_a, and check whether a singleton
            v_a = self.Zv[col] 

            # get CRP probability of column in current view
            if is_singleton:
                pv = log(float(self.alpha))
            else:
                pv = log(float(self.Nv[v_a]-1))

            # log p for original assignment
            p_v_a = dim.full_marginal_logp()+pv


            # Metropolis move on birth-death. Decide whether to propose new or to shift.
            p_new_propose_new_view = 0.5
            create_new = random.random() < p_new_propose_new_view

            # begin calculating transition probability (not always symmetric)
            p_a_to_b = log(p_new_propose_new_view)  # P(v_a -> v_b)
            p_b_to_a = log(p_new_propose_new_view)  # P(v_b -> v_a)

            if create_new:
                # Propose new view
                v_b = self.V
                proposal_view = cc_view([dim], n_grid=self.n_grid)

                # calculate probability under new view
                dim.reassign(proposal_view.Z)
                p_v_b = dim.full_marginal_logp()+log(float(self.alpha))
                
                # If v_a is not a singleton then the proposals probabilities will be 
                # different because a view is created if v_a assigned to v_b.
                # v_a->v_b happens if create_new: P(v_a->v_b) = .5*1
                # v_b->v_a happens if not create_new and v_a selected: P(v_b->v_a) = .5*1/(V+1)
                if not is_singleton:
                    p_b_to_a += log( 1.0/(self.V+1) )

                # MH jump
                if log(random.random()) < (p_v_b-p_v_a + p_b_to_a - p_a_to_b):
                    if is_singleton:
                        self.__swap_singleton_views(dim, v_a, proposal_view)
                    else:
                        self.__create_singleton_view(dim, v_a, proposal_view)
                else:
                    self.dims[col].reassign(self.views[v_a].Z)
            else:
                # choose a random view, v_b, to jump to
                v_b = random.randrange(self.V)

                # we only need to do something if v_a != v_b
                if v_b != v_a:
                    # calculate the probability under the new view
                    dim.reassign(self.views[v_b].Z)
                    p_v_b = dim.full_marginal_logp()+log(float(self.Nv[v_b]))

                    # if v_a is a singleton, the proposal probabilities are not identical
                    # because v_b would be moving to a singleton
                    # v_a->v_b if not create_new and v_b selected P(v_a->v_b) = .5*1/V
                    # v_b->v_a if create_new P(v_b->v_a) = .5*1
                    if is_singleton:
                        p_a_to_b += log(1.0/self.V)

                    # MH jump
                    if log(random.random()) < (p_v_b-p_v_a + p_b_to_a - p_a_to_b):
                        if is_singleton:
                            self.__destroy_singleton_view(dim, v_a, v_b)
                        else:
                            self.__move_dim_to_view(dim, v_a, v_b)
                    else:
                        self.dims[col].reassign(self.views[v_a].Z)

            # self.__check_partitions()

    def __transition_columns_kernel_2(self, which_cols=None, m=3, append=False):
        """ column assignment transition kernel 2: Gibbs with auxiliary parameters"""

        if append:
            which_cols = [self.n_cols-1]

        if which_cols is None:
            which_cols = [i for i in range(self.n_cols)]

        random.shuffle(which_cols)

        for col in which_cols:
            # get start view, v_a, and check whether a singleton
            v_a = self.Zv[col]

            if append:
                is_singleton = False
                pv = list(self.Nv)
            else:
                is_singleton = (self.Nv[v_a] == 1)

                pv = list(self.Nv)
                # Get crp probabilities under each view. remove from current view.
                # If v_a is a singleton, do not consider move to new singleton view.
                if is_singleton:
                    pv[v_a] = self.alpha
                else:
                    pv[v_a] -= 1

            # take the log
            pv = numpy.log(numpy.array(pv))

            ps = []
            # calculate probability under each view's assignment
            dim = self.dims[col]
            for v in range(self.V):
                dim.reassign(self.views[v].Z)
                p_v = dim.full_marginal_logp()+pv[v]
                ps.append(p_v)

            # if not a singleton, propose m auxiliary parameters (views)
            if not is_singleton:
                # crp probability of singleton, split m times.
                log_aux = log(self.alpha/float(m))
                proposal_views = []
                for  _ in range(m):
                    # propose (from prior) and calculate probability under each view
                    proposal_view = cc_view([dim], n_grid=self.n_grid)
                    proposal_views.append(proposal_view)
                    dim.reassign(proposal_view.Z)
                    p_v = dim.full_marginal_logp()+log_aux
                    ps.append(p_v)

            # draw a view
            v_b = utils.log_pflip(ps)

            if append:
                if v_b >= self.V:
                    index = v_b-self.V
                    assert( index >= 0 and index < m)
                    proposal_view = proposal_views[index]
                self.__append_new_dim_to_view(dim, v_b, proposal_view)
                continue

            # clean up
            if v_b != v_a:
                if is_singleton:
                    assert( v_b < self.V )
                    self.__destroy_singleton_view(dim, v_a, v_b)
                elif v_b >= self.V:
                    index = v_b-self.V
                    assert( index >= 0 and index < m)
                    proposal_view = proposal_views[index]
                    self.__create_singleton_view(dim, v_a, proposal_view)
                else:
                    self.__move_dim_to_view(dim, v_a, v_b)
            else:
                self.dims[col].reassign(self.views[v_a].Z)

            # for debugging
            self.__check_partitions()

    def __transition_rows(self, which_rows=None):
        # move rows to new cluster
        for view in self.views:
            view.reassign_rows_to_cats(which_rows=which_rows)
    
    def __transition_column_hypers(self, which_cols=None):
        if which_cols is None:
            which_cols = range(self.n_cols)

        for i in which_cols:
            self.dims[i].update_hypers()

    def __transition_view_alphas(self):
        for view in self.views:
            view.transition_alpha()

    def __transition_state_alpha(self):
        
        logps = numpy.zeros(self.n_grid)
        for i in range(self.n_grid):
            alpha = self.alpha_grid[i]
            logps[i] = utils.unorm_lcrp_post(alpha, self.n_cols, self.V, lambda x: 0)
        # log_pdf_lambda = lambda a : utils.lcrp(self.n_cols, self.Nv, a) + self.alpha_prior_lambda(a)
        
        index = utils.log_pflip(logps)
        self.alpha = self.alpha_grid[index]

    def __destroy_singleton_view(self, dim, to_destroy, move_to):
        self.Zv[dim.index] = move_to
        self.views[to_destroy].release_dim(dim.index)   
        zminus = numpy.nonzero(self.Zv>to_destroy)
        self.Zv[zminus] -= 1
        self.views[move_to].assimilate_dim(dim)
        self.Nv[move_to] += 1
        del self.Nv[to_destroy]
        del self.views[to_destroy]
        self.V -= 1

    def __swap_singleton_views(self, dim, view_index, proposal_view):
        self.views[view_index] = proposal_view
        dim.reassign(proposal_view.Z)

    def __create_singleton_view(self, dim, current_view_index, proposal_view):
        self.Zv[dim.index] = self.V
        dim.reassign(proposal_view.Z)
        self.views[current_view_index].release_dim(dim.index)
        self.Nv[current_view_index] -= 1
        self.Nv.append(1)
        self.views.append(proposal_view)
        self.V += 1

    def __move_dim_to_view(self, dim, move_from, move_to):
        self.Zv[dim.index] = move_to
        self.views[move_from].release_dim(dim.index)
        self.Nv[move_from] -= 1
        self.views[move_to].assimilate_dim(dim)
        self.Nv[move_to] += 1

    def __append_new_dim_to_view(self, dim, append_to, proposal_view):
        self.Zv[dim.index] = append_to
        if append_to == self.V:
            self.Nv.append(1)
            self.V += 1
            self.views.append(proposal_view)
        else:
            self.Nv[append_to] += 1
            self.views[append_to].assimilate_dim(dim)

        self.__check_partitions()

    def __plot(self, fig, layout):
        # do not plot more than 6 by 4
        if self.n_cols > 24:
            return

        pylab.clf()
        for dim in self.dims:
            index = dim.index
            ax = pylab.subplot(layout['plots_x'], layout['plots_y'], index)
            if self.Zv[index] >= len(layout['border_color']):
                border_color = 'gray'
            else:
                border_color = layout['border_color'][self.Zv[index]]
            dim.plot_dist()
            ax.spines['bottom'].set(lw=7,color=border_color)
            ax.spines['top'].set(lw=7,color=border_color)
            ax.spines['right'].set(lw=7,color=border_color)
            ax.spines['left'].set(lw=7,color=border_color)
            pylab.text(1,1, "K: %i " % len(dim.clusters),
                transform=ax.transAxes,
                fontsize=18, weight='bold', color='blue',
                horizontalalignment='right',verticalalignment='top')

        pylab.draw()

    # for debugging only
    def __check_partitions(self):
        # Nv should account for each column
        assert sum(self.Nv) == self.n_cols

        # Nv should have an entry for each view
        assert len(self.Nv) == self.V

        #
        assert max(self.Zv) == self.V-1

        for v in range(len(self.Nv)):
            # check that the number of dims actually assigned to the view matches 
            # the count in Nv
            assert len(self.views[v].dims) == self.Nv[v]

            Nk = self.views[v].Nk
            K = self.views[v].K

            assert sum(Nk) == self.n_rows
            assert len(Nk) == K
            assert max(self.views[v].Z) == K-1

            for dim in self.views[v].dims.values():
                # make sure the number of clusters in each dim in the view is the same
                # and is the same as described in the view (K, Nk)
                assert len(dim.clusters) == len(Nk)
                assert len(dim.clusters) == K
                for k in range(len(dim.clusters)):
                    assert dim.clusters[k].N == Nk[k]



