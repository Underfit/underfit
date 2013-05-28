'''
adapted from https://github.com/Theano/Theano/wiki/Cookbook
'''

from scipy import stats
import theano
import theano.compile.mode
import theano.scalar
import theano.gof

# This compile mode will optimize the expression tree but will not compile any of the code.
_FAST_PY = theano.compile.mode.Mode('py', 'fast_run')

def _is_symbolic(v, kw):
    r"""Return `True` if any of the arguments are symbolic."""
    symbolic = False
    v = list(v)
    for _container, _iter in [(v, xrange(len(v))), (kw, kw)]:
        for _k in _iter:
            _v = _container[_k]
            if isinstance(_v, theano.gof.Variable):
                symbolic = True
    return symbolic

class Chi2PV(theano.scalar.ScalarOp):
    r"""Theano wrapper for :func:`Chi2PV`"""
    def __init__(self):
        self.name = "Chi2PV"
        self.output_types_preference = theano.scalar.upgrade_to_float

    def impl(self, x, k):
        r"""Just compute the numerical result here."""
        
        return stats.chi2.sf(x, k)
    
    '''
    def grad(self, (alpha, x), (gz,)):
        r"""Return the derivative.  Note: I don't the derivative with respect to alpha
        so I return `None`.  Note also that you must also multiply by `gz` here to
        carry through the chain rule as specified in the documentation."""
        dP_dx = (alpha + 1.0)*(x*LegendreP()(alpha, x)
                                - LegendreP()(1.0 + alpha, x))/(1.0 - x*x)
        return [None, gz * dP_dx]
    '''
    
class Elemwise(theano.tensor.Elemwise):
    r"""Wrapper for :class:`theano.tensor.Elemwise` that overloads
    :meth:`__call__` to directly call the implementation for numerical
    arguments."""
    def __call__(self, *v, **kw):
        if _is_symbolic(v, kw):
            return theano.tensor.Elemwise.__call__(self, *v, **kw)
        else:
            return self.scalar_op.impl(*v, **kw)


scalar_chi2pv = Chi2PV()
chi2pv = Elemwise(scalar_chi2pv)
