"""
Change-point optimizers for the five-parameter change-point model.

All optimizers estimate the heating and cooling change-points (T_h, T_c) of

    E = E_b + beta_h * max(0, T_h - T) + beta_c * max(0, T - T_c)

from a ``cpm.dataset.Dataset``. Optimization is carried out on standardized
temperature and energy; for given change-points, the remaining parameters
(E_b, beta_h, beta_c) are obtained by linear regression and converted back to
the original units.

Implemented optimizers (class name without the ``Optimizer`` suffix):

- ``Reference``: reference change-points (no optimization).
- ``IMT``: two-stage grid search of the Inverse Modeling Toolkit (Kissock et al., 2003).
- ``GridSearch``: single-stage grid search with a fixed step.
- ``TRR``: Trust Region Reflective least squares from a random start.
- ``MultiStartTRR``: TRR from several random starts.
- ``InformedStartTRR``: TRR from a start informed by the population distribution
  of the reference change-points (i.e. with information leakage).
- ``Muggeo``: Muggeo's iterative piecewise regression (``piecewise_regression``).
- ``SegmentedRegression``: exhaustive search over splits of the sorted data.
"""

import dataclasses as dc
import functools
import sys
import time
from typing import TYPE_CHECKING, ClassVar, Literal

import more_itertools as mi
import numpy as np
import piecewise_regression as pr
import statsmodels.api as sm
from scipy import optimize as opt
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike
    from statsmodels.regression.linear_model import RegressionResultsWrapper

    from cpm import dataset

type LinearModel = Literal['OLS', 'HuberT']
"""Linear model for E_b, beta_h and beta_c: OLS or robust (Huber's T) regression."""

type ObjectFunction = Literal['SSE', 'HuberT']
"""Objective function of the change-point search: SSE or Huber loss of residuals."""

OPTIMIZERS = (
    'Reference',
    'IMT',
    'IMT(Huber)',
    'GS',
    'SR',
    'Muggeo',
    'TRR(Linear)',
    'TRR(SoftL1)',
    'TRR(MS)',
    'TRR(IS)',
)
"""Display names of the optimizers, in the order used in tables and figures."""

OPTIMIZERS_ORDER = {x: i for i, x in enumerate(OPTIMIZERS)}

_HUBER_T = sm.robust.norms.HuberT()


def _metrics(ytrue: np.ndarray, ypred: np.ndarray, p: int = 5):
    """
    Goodness-of-fit metrics of a fitted model.

    RMSE, CV(RMSE) and NMBE use ``n - p`` degrees of freedom following
    ASHRAE Guideline 14.

    Parameters
    ----------
    ytrue
        Observed energy use.
    ypred
        Predicted energy use.
    p
        Number of model parameters.

    Returns
    -------
    dict of str to float
        ``metric.r2``, ``metric.sse``, ``metric.rmse``, ``metric.cvrmse``,
        ``metric.nmbe``, ``metric.mape`` and ``metric.mbe``. Both
        ``metric.nmbe`` and ``metric.mbe`` are based on ``ypred - ytrue``,
        i.e. the opposite sign of the NMBE in ASHRAE Guideline 14.
    """
    dof = ytrue.size - p
    error = ypred - ytrue
    ybar = np.mean(ytrue)

    nmbe = np.sum(error) / (dof * ybar)
    sse = np.sum(error**2)
    rmse = np.sqrt(sse / dof)

    return {
        'metric.r2': metrics.r2_score(ytrue, ypred),
        'metric.sse': sse,
        'metric.rmse': rmse,
        'metric.cvrmse': rmse / ybar,
        'metric.nmbe': nmbe,
        'metric.mape': metrics.mean_absolute_percentage_error(ytrue, ypred),
        'metric.mbe': np.mean(error),
    }


class ConvergenceError(RuntimeError):
    """Raised when an optimizer fails to find a valid solution."""


@dc.dataclass
class BaseOptimizer:
    """
    Base class of the change-point optimizers.

    Subclasses implement ``optimize``, which returns the change-points in
    standardized temperature units. Calling the optimizer runs ``optimize``,
    refits the linear model at the returned change-points, and reports the
    estimated parameters, goodness-of-fit metrics and diagnostics.

    Parameters
    ----------
    dataset
        Dataset to fit.
    linear_model
        Linear model used to estimate (E_b, beta_h, beta_c) for given
        change-points.
    object
        Objective function minimized over the change-points.
    xlim
        Admissible range of the change-points [°C]. Solutions with
        T_h <= xlim[0] or T_c >= xlim[1] are reported as not converged.

    Attributes
    ----------
    P
        Number of model parameters, used for the degrees of freedom of the
        metrics.
    """

    dataset: dataset.Dataset
    linear_model: LinearModel = 'OLS'
    object: ObjectFunction = 'SSE'

    xlim: tuple[float, float] = (0, 25)  # °C
    # NOTE: Kissock et al. (2003) suggested 41-80°F,
    # which differs from the results for Korean buildings

    _count_lm_fit: int = 0
    _count_obj_eval: int = 0
    _xscaler: StandardScaler = dc.field(default_factory=StandardScaler)
    _yscaler: StandardScaler = dc.field(default_factory=StandardScaler)
    _xscaled: np.ndarray = dc.field(init=False)
    _yscaled: np.ndarray = dc.field(init=False)

    P: ClassVar[int] = 5  # number of parameters

    def __post_init__(self):
        x = self.dataset.temperature
        y = self.dataset.energy
        self._xscaled = self._xscaler.fit_transform(x.reshape(-1, 1)).ravel()
        self._yscaled = self._yscaler.fit_transform(y.reshape(-1, 1)).ravel()

    def name(self):
        """
        Return the optimizer name: class name without ``Optimizer``.

        Returns
        -------
        str
        """
        return self.__class__.__name__.removesuffix('Optimizer')

    @functools.cached_property
    def _linear_model_criterion(self):
        """Robust norm of ``linear_model``, or None for OLS."""
        match self.linear_model:
            case 'OLS':
                return None
            case 'HuberT':
                return _HUBER_T

        raise ValueError(self.linear_model)

    @property
    def _xscaler_std(self) -> float:
        """Standard deviation of the temperature used for scaling [°C]."""
        return self._xscaler.scale_[0]  # ty: ignore[not-subscriptable]

    def hyper_params(self):
        """
        Return the public hyperparameters of the optimizer.

        Returns
        -------
        dict
            Public fields except ``dataset``, with keys prefixed by ``params.``.
        """
        d = dc.asdict(self)
        d.pop('dataset')
        return {f'params.{k}': v for k, v in d.items() if not k.startswith('_')}

    def _xscale(self, x: ArrayLike):
        """
        Standardize temperature value(s).

        Parameters
        ----------
        x
            Temperature [°C].

        Returns
        -------
        numpy.ndarray
            Standardized temperature, flattened.
        """
        return self._xscaler.transform(np.array(x).reshape(-1, 1)).ravel()

    def fit_linear_model(
        self,
        scp: np.ndarray,
        *,
        count: bool = True,
    ) -> RegressionResultsWrapper:
        """
        Fit the linear model for given change-points.

        Parameters
        ----------
        scp
            Standardized change-points (T_h, T_c).
        count
            Whether to add to the linear-model fit counter. For robust
            regression, the number of IRLS iterations is added.

        Returns
        -------
        statsmodels.regression.linear_model.RegressionResultsWrapper
            Fitted model in standardized units; ``params`` are
            (E_b, beta_h, beta_c).
        """
        hdd = np.maximum(0, scp[0] - self._xscaled)
        cdd = np.maximum(0, self._xscaled - scp[1])
        x = np.column_stack([np.ones_like(self._xscaled), hdd, cdd])

        if (m := self._linear_model_criterion) is None:
            model = sm.OLS(endog=self._yscaled, exog=x).fit()
        else:
            model = sm.RLM(endog=self._yscaled, exog=x, M=m).fit()

        if count:
            self._count_lm_fit += (
                1
                if self._linear_model_criterion is None
                else model.fit_history['iteration']
            )

        # ``model.params`` = (baseline, beta_h, beta_c)
        return model

    def residuals(self, scp: np.ndarray) -> np.ndarray:
        """
        Residuals of the linear model for given change-points.

        Increments the objective evaluation counter.

        Parameters
        ----------
        scp
            Standardized change-points (T_h, T_c).

        Returns
        -------
        numpy.ndarray
            Residuals in standardized units.
        """
        self._count_obj_eval += 1
        model = self.fit_linear_model(scp)
        return model.resid

    def object_function(self, scp: np.ndarray) -> float:
        """
        Objective value for given change-points.

        Parameters
        ----------
        scp
            Standardized change-points (T_h, T_c).

        Returns
        -------
        float
            Sum of squared residuals (``'SSE'``) or sum of Huber's rho of
            the residuals (``'HuberT'``).

        Raises
        ------
        ValueError
            If ``object`` is unknown.
        """
        resid = self.residuals(scp)

        match self.object:
            case 'SSE':
                return np.sum(np.square(resid))
            case 'HuberT':
                return np.sum(_HUBER_T.rho(resid))
            case _:
                raise ValueError(self.object)

    def _model(self, scp: np.ndarray):
        """
        Model parameters and metrics in original units.

        Parameters
        ----------
        scp
            Standardized change-points (T_h, T_c).

        Returns
        -------
        dict
            Estimated parameters (``<name>.pred``), R² of the linear model
            (``metric.ols.*``; None for robust regression) and the metrics of
            ``_metrics``.
        """
        # change-points
        cp = self._xscaler.inverse_transform(scp.reshape(-1, 1)).ravel()

        # baseline
        model = self.fit_linear_model(scp, count=False)
        baseline = self._yscaler.inverse_transform([[model.params[0]]]).item()

        # beta
        r = np.sqrt(self._yscaler.var_[0] / self._xscaler.var_[0])  # std_y / std_x  # ty:ignore[not-subscriptable]
        beta_h = model.params[1] * r
        beta_c = model.params[2] * r

        # SSE
        ypred = (
            baseline
            + beta_h * np.maximum(0, cp[0] - self.dataset.temperature)
            + beta_c * np.maximum(0, self.dataset.temperature - cp[1])
        )
        ytrue = self.dataset.energy

        return {
            'baseline.pred': baseline,
            'beta_h.pred': beta_h,
            'beta_c.pred': beta_c,
            't_h.pred': cp[0],
            't_c.pred': cp[1],
            'metric.ols.r2': getattr(model, 'rsquared', None),
            'metric.ols.r2adj': getattr(model, 'rsquared_adj', None),
            **_metrics(ytrue, ypred, p=self.P),
        }

    def optimize(self) -> np.ndarray | tuple[np.ndarray, dict | None]:
        """
        Search the change-points.

        Returns
        -------
        numpy.ndarray or tuple of (numpy.ndarray, dict or None)
            Standardized change-points (T_h, T_c), optionally with additional
            diagnostics to be included in the results.

        Raises
        ------
        ConvergenceError
            If no valid solution is found.
        """
        raise NotImplementedError

    def __call__(self):
        """
        Run the optimizer and collect the results.

        Returns
        -------
        dict
            Optimizer name and hyperparameters, dataset statistics, estimated
            parameters and metrics, additional diagnostics, convergence flag,
            error code, run time [s] and evaluation counts. The error code is
            ``'Numerical'`` (optimizer failure), ``'CP'`` (T_h >= T_c),
            ``'Bound.Th'``/``'Bound.Tc'`` (change-point at or beyond ``xlim``),
            or an empty string if converged. Run time is None if not
            converged.
        """
        start = time.perf_counter()
        end = 0.0

        try:
            res = self.optimize()
        except ConvergenceError:
            model = {}
            additional = None
            converged = False
            error = 'Numerical'
        else:
            end = time.perf_counter()
            if isinstance(res, tuple):
                scp, additional = res
            else:
                scp = res
                additional = None

            model = self._model(scp)

            if model['t_h.pred'] >= model['t_c.pred']:
                converged = False
                error = 'CP'
            elif model['t_h.pred'] <= self.xlim[0]:
                converged = False
                error = 'Bound.Th'
            elif model['t_c.pred'] >= self.xlim[1]:
                converged = False
                error = 'Bound.Tc'
            else:
                converged = True
                error = ''

        return {
            'class': self.__class__.__name__,
            'optimizer': self.name(),
            **self.hyper_params(),
            **dict(self.dataset.stats()),
            **model,
            **(additional or {}),
            'converged': converged,
            'error': error,
            'duration': end - start if converged else None,
            'count.lm-fit': self._count_lm_fit,
            'count.obj-eval': self._count_obj_eval,
        }


@dc.dataclass
class ReferenceOptimizer(BaseOptimizer):
    """
    Reference model fitted at the reference change-points.

    Only (E_b, beta_h, beta_c) are estimated, so ``P`` is 3.
    """

    P: ClassVar[int] = 3

    def __call__(self):
        """
        Fit the linear model at the reference change-points.

        Returns
        -------
        dict
            Optimizer name, dataset statistics, estimated parameters and
            metrics; always converged.
        """
        cp = np.array([self.dataset.t_h, self.dataset.t_c])
        scp = self._xscaler.transform(cp.reshape(-1, 1)).ravel()
        model = self._model(scp)
        return {
            'class': self.__class__.__name__,
            'optimizer': self.name(),
            **dict(self.dataset.stats()),
            **model,
            'converged': True,
        }


@dc.dataclass
class _GridSearchOptimizer(BaseOptimizer):
    """Base class of the grid search optimizers (``scipy.optimize.brute``)."""

    @functools.cached_property
    def scaled_xlim(self):
        """``xlim`` in standardized units."""
        return self._xscale(self.xlim)

    def object_function(self, scp: np.ndarray):
        """
        Objective value; infinite unless T_h < T_c.

        Parameters
        ----------
        scp
            Standardized change-points (T_h, T_c).

        Returns
        -------
        float
        """
        if scp[0] + 1e-8 >= scp[1]:
            return np.inf

        return super().object_function(scp)

    def _optimize(
        self,
        step: float,
        ranges: tuple[slice, slice] | None = None,
    ) -> np.ndarray:
        """
        Evaluate the objective on a grid and return its minimizer.

        Parameters
        ----------
        step
            Grid step [°C]. Used only if ``ranges`` is None, in which case both
            change-points are searched over ``xlim`` with this step.
        ranges
            Search ranges of (T_h, T_c) in standardized units, passed to
            ``scipy.optimize.brute``.

        Returns
        -------
        numpy.ndarray
            Standardized change-points (T_h, T_c).
        """
        if ranges is not None:
            ranges_ = ranges
        else:
            n = int(np.floor((self.xlim[1] - self.xlim[0]) / step + 1e-9)) + 1
            s = slice(self.scaled_xlim[0], self.scaled_xlim[1], complex(0, n))
            ranges_ = (s, s)

        return opt.brute(
            func=self.object_function,
            ranges=ranges_,
            finish=None,
        )


@dc.dataclass
class IMTOptimizer(_GridSearchOptimizer):
    """
    Two-stage grid search of the Inverse Modeling Toolkit (IMT).

    The first stage searches ``xlim`` with ``steps[0]``; each subsequent stage
    searches ±(previous step) around the previous solution with the next
    step.

    Parameters
    ----------
    steps
        Grid step of each stage [°C].
    """

    # two-stage grid search (Kissock et al., 2003)
    # first step = (Xmax - Xmin)/10
    steps: Sequence[float] = (2.5, 0.25)

    def name(self):
        """Return ``IMT`` followed by the final step, e.g. ``IMT0.25``."""
        return f'{super().name()}{self.steps[-1]}'

    def optimize(self):
        """
        Run the multi-stage grid search.

        Returns
        -------
        numpy.ndarray
            Standardized change-points (T_h, T_c).
        """
        lx, ux = self.scaled_xlim
        scale = self._xscaler_std
        res = self._optimize(step=self.steps[0])
        for s1, s2 in mi.windowed(self.steps, n=2):
            assert s1 is not None
            assert s2 is not None

            ss1 = s1 / scale
            ss2 = s2 / scale
            ranges = (
                slice(max(lx, res[0] - ss1), res[0] + ss1, ss2),
                slice(res[1] - ss1, min(ux, res[1] + ss1), ss2),
            )
            res = self._optimize(step=s2, ranges=ranges)

        return res


@dc.dataclass
class GridSearchOptimizer(_GridSearchOptimizer):
    """
    Single-stage grid search.

    Parameters
    ----------
    step
        Grid step [°C].
    """

    step: float = 0.15

    def name(self):
        """Return ``GridSearch`` followed by the step, e.g. ``GridSearch0.25``."""
        return f'{super().name()}{self.step}'

    def optimize(self):
        """
        Run the grid search.

        Returns
        -------
        numpy.ndarray
            Standardized change-points (T_h, T_c).
        """
        return self._optimize(step=self.step)


@dc.dataclass
class _NumericalOptimizer(BaseOptimizer):
    """
    Base class of the optimizers that start from an initial guess.

    The initial guess is drawn uniformly from ``TH0`` and ``TC0`` with a random
    number generator seeded by the station, year and repetition of the
    dataset.

    Attributes
    ----------
    TH0, TC0
        Ranges of the initial T_h and T_c [°C].
    """

    TH0: ClassVar[tuple[float, float]] = (0, 12)
    TC0: ClassVar[tuple[float, float]] = (13, 25)

    @functools.cached_property
    def rng(self):
        """Random number generator seeded by (station, year, repeat)."""
        return np.random.default_rng([
            self.dataset.station,
            self.dataset.year,
            self.dataset.repeat,
        ])

    @functools.cached_property
    def x0(self):
        """Initial guess (T_h, T_c) [°C]."""
        return [self.rng.uniform(*self.TH0), self.rng.uniform(*self.TC0)]


@dc.dataclass
class TRROptimizer(_NumericalOptimizer):
    """
    Trust-Region Reflective least squares (``scipy.optimize.least_squares``).

    The change-points are reparameterized as the midpoint and half-gap,
    ``(T_h, T_c) = (T_mid - Δ, T_mid + Δ)``, so that T_h < T_c is enforced by
    the bound Δ >= 0.125 °C (a minimum gap of 0.25 °C, equal to the final IMT
    step).

    Parameters
    ----------
    loss
        Loss function of ``scipy.optimize.least_squares``.
    """

    loss: Literal['linear', 'soft_l1'] = 'linear'

    def name(self):
        """Return ``TRR`` followed by the loss, e.g. ``TRRLinear``."""
        loss = 'Linear' if self.loss == 'linear' else 'SoftL1'
        return f'{super().name()}{loss}'

    def residuals(self, scp: np.ndarray):
        """
        Residuals for reparameterized change-points.

        Parameters
        ----------
        scp
            Standardized (T_mid, Δ).

        Returns
        -------
        numpy.ndarray
        """
        return super().residuals(np.array([scp[0] - scp[1], scp[0] + scp[1]]))

    def _optimize(self, x0: np.ndarray):
        """
        Run TRR from an initial guess.

        Parameters
        ----------
        x0
            Initial guess (T_h, T_c) [°C].

        Returns
        -------
        numpy.ndarray
            Standardized change-points (T_h, T_c).

        Raises
        ------
        ConvergenceError
            If ``scipy.optimize.least_squares`` does not succeed.
        """
        scale = self._xscaler_std

        # scaled params
        sx0 = self._xscale(x0)
        sg0 = (x0[1] - x0[0]) / (2 * scale)

        sxlim = self._xscale(self.xlim)
        bounds = (
            [sxlim[0], 0.125 / scale],  # lower bound of a 0.25℃ gap, same as IMT
            [sxlim[1], (self.xlim[1] - self.xlim[0]) / (2 * scale)],
        )

        r = opt.least_squares(
            fun=self.residuals,
            x0=np.array([sx0.mean(), sg0]),
            bounds=bounds,
            method='trf',
            x_scale='jac',
            loss=self.loss,
        )
        if not r.success:
            raise ConvergenceError

        return np.array([r.x[0] - r.x[1], r.x[0] + r.x[1]])

    def optimize(self):
        """
        Run TRR from ``x0``.

        Returns
        -------
        numpy.ndarray
            Standardized change-points (T_h, T_c).
        dict
            Initial guess (``params.Th0``, ``params.Tc0``).
        """
        return (
            self._optimize(self.x0),
            {'params.Th0': self.x0[0], 'params.Tc0': self.x0[1]},
        )


@dc.dataclass
class MultiStartTRROptimizer(TRROptimizer):
    """
    TRR from several random initial guesses.

    Among the starts that converge within ``xlim``, the solution with the
    smallest objective value is selected. Only the SSE objective is
    supported.

    Parameters
    ----------
    loss
        Loss function of ``scipy.optimize.least_squares``.
    multi_start
        Number of initial guesses.
    """

    loss: Literal['linear', 'soft_l1'] = 'linear'
    multi_start: int = 5

    def _iter(self):
        """
        Run TRR from each initial guess.

        Yields
        ------
        numpy.ndarray
            Standardized change-points of each start that converged with
            both change-points strictly inside ``xlim``.
        """
        x0array = np.array([
            self.rng.uniform(*self.TH0, size=self.multi_start),
            self.rng.uniform(*self.TC0, size=self.multi_start),
        ]).T
        xlim = self._xscale(np.array(self.xlim).reshape(-1, 1)).ravel()

        for x0 in x0array:
            try:
                scp = self._optimize(x0)
            except ConvergenceError:
                continue

            if scp[0] <= xlim[0] or xlim[1] <= scp[1]:
                continue

            yield scp

    def optimize(self):
        """
        Select the best solution among the starts.

        Returns
        -------
        numpy.ndarray
            Standardized change-points (T_h, T_c).
        dict
            Variance of the objective values across the valid starts
            (``trr.sse_var``).

        Raises
        ------
        ConvergenceError
            If no start yields a valid solution.
        """
        scp = list(self._iter())

        if not scp:
            raise ConvergenceError

        sse = [self.object_function(x) for x in scp]
        idx = np.argmin(sse)

        return scp[idx], {'trr.sse_var': np.var(sse)}

    def __call__(self):
        """
        Run the optimizer; see ``BaseOptimizer.__call__``.

        Returns
        -------
        dict
            Results of ``BaseOptimizer.__call__``.

        Raises
        ------
        ValueError
            If ``object`` is not ``'SSE'``.
        """
        if self.object != 'SSE':
            raise ValueError(self.object)

        return super().__call__()


@dc.dataclass
class InformedStartTRROptimizer(TRROptimizer):
    """
    TRR with the initial guess informed by the reference change-points.

    T_h and T_c are drawn from ±2°C around 11.0°C and 16.1°C, the means of
    the distribution from which the reference change-points are sampled.
    """

    TH0: ClassVar[tuple[float, float]] = (9, 13)
    TC0: ClassVar[tuple[float, float]] = (14.1, 18.1)


@dc.dataclass
class MuggeoOptimizer(_NumericalOptimizer):
    """
    Muggeo's iterative method for segmented regression.

    A continuous three-segment piecewise-linear model is fitted with
    ``piecewise_regression.Fit``, including bootstrap restarting. The slope of
    the middle segment is not constrained to zero; the 5P linear model is
    refitted at the estimated breakpoints.
    """

    def optimize(self):
        """
        Run Muggeo's method from ``x0``.

        Returns
        -------
        numpy.ndarray
            Standardized change-points (T_h, T_c), sorted.
        dict
            Initial guess, slope of the middle (baseline) segment in original
            units (``muggeo.beta_baseline``), and the number of iterations of
            the best fit and of bootstrap restarts.

        Raises
        ------
        ConvergenceError
            If ``piecewise_regression`` does not converge.
        """
        x0 = self._xscale(self.x0)
        min_gap = 0.25 / self._xscaler_std / np.ptp(self._xscaled)  # 0.25℃
        fit = pr.Fit(
            xx=self._xscaled,
            yy=self._yscaled,
            start_values=x0,
            # SD(T)=9~11℃, SD(xscaled)=1
            # tolerance = 0.001 SD(x) -> approx. 0.01℃
            tolerance=0.001,
            min_distance_between_breakpoints=min_gap,
            min_distance_to_edge=1e-8,
        )
        params = fit.get_params()

        if not params['converged']:
            raise ConvergenceError

        assert fit.best_muggeo is not None
        # fits on bootstrap-resampled data cannot be counted
        self._count_lm_fit = sum(len(m.fit_history) for m in fit.bootstrap_history)

        cp = np.sort([params['breakpoint1'], params['breakpoint2']])
        beta_baseline = (
            params['alpha2'] * self._yscaler.scale_[0] / self._xscaler.scale_[0]  # ty: ignore[not-subscriptable]
        )
        res = {
            'params.Th0': self.x0[0],
            'params.Tc0': self.x0[1],
            'muggeo.beta_baseline': beta_baseline,
            'muggeo.count.best_fit': len(fit.best_muggeo.fit_history),
            'muggeo.count.bootstrapping': len(fit.bootstrap_history),
        }
        return cp, res


@dc.dataclass(frozen=True)
class _SegmentedParams:
    """
    OLS result of a single segment, ``y = alpha + beta * x``.

    Parameters
    ----------
    alpha
        Intercept.
    beta
        Slope.
    sse
        Sum of squared residuals.
    """

    alpha: float
    beta: float
    sse: float


@dc.dataclass(frozen=True)
class _SegmentedData:
    """
    Sorted data with cumulative sums for O(1) segment regression.

    Each cumulative-sum array has a leading zero, so the sum over the
    samples ``[start, end)`` is ``s[end] - s[start]``.

    Parameters
    ----------
    xx, yy
        Standardized temperature and energy, sorted by temperature.
    sx, sy, sxx, syy, sxy
        Cumulative sums of x, y, x², y² and xy.
    """

    xx: np.ndarray
    yy: np.ndarray
    sx: np.ndarray
    sy: np.ndarray
    sxx: np.ndarray
    syy: np.ndarray
    sxy: np.ndarray

    @functools.cached_property
    def n(self):
        """Number of samples."""
        return self.xx.size

    def ols(self, start: int, end: int, threshold: float = 1e-12):
        """
        Fit a straight line to the samples ``[start, end)``.

        Parameters
        ----------
        start, end
            Sample range (end exclusive).
        threshold
            If the absolute determinant of the normal equations is below this
            value, the segment is treated as degenerate.

        Returns
        -------
        _SegmentedParams
            Fitted line; all zeros if the segment is degenerate.
        """
        size = end - start
        sx = self.sx[end] - self.sx[start]
        sy = self.sy[end] - self.sy[start]
        sxx = self.sxx[end] - self.sxx[start]
        syy = self.syy[end] - self.syy[start]
        sxy = self.sxy[end] - self.sxy[start]

        denom = size * sxx - sx**2
        if abs(denom) < threshold:
            return _SegmentedParams(0, 0, 0)

        beta = (size * sxy - sx * sy) / denom
        alpha = (sy - beta * sx) / size
        sse = syy - alpha * sy - beta * sxy
        return _SegmentedParams(alpha, beta, sse)


@dc.dataclass(frozen=True)
class _BestCase:
    """
    Best split found by ``SegmentedRegressionOptimizer``.

    Parameters
    ----------
    sse
        Total SSE of the three segments.
    th, tc
        Standardized change-points.
    heating_end
        Exclusive end index of the heating segment.
    cooling_start
        Start index of the cooling segment.
    """

    sse: float = np.inf
    th: float = np.nan
    tc: float = np.nan
    heating_end: int = -1
    cooling_start: int = -1


@dc.dataclass
class SegmentedRegressionOptimizer(BaseOptimizer):
    """
    Exhaustive search over splits of the data sorted by temperature.

    For every split into heating, baseline and cooling segments, a line is
    fitted to the heating and cooling segments and a constant to the
    baseline segment. Splits with a non-negative heating slope or a
    non-positive cooling slope are skipped. The change-points are the
    intersections of the heating and cooling lines with the baseline, and the
    split with the smallest total SSE that satisfies the constraints is
    selected.

    Parameters
    ----------
    min_gap
        Minimum gap between T_h and T_c [°C].
    """

    min_gap: float = 0.25

    def min_samples(self, threshold: int = 50):
        """
        Minimum number of samples per segment.

        Parameters
        ----------
        threshold
            Dataset size below which fewer samples are required.

        Returns
        -------
        int
            2 if the dataset has fewer than ``threshold`` samples (monthly dataset),
            otherwise 3 (daily dataset).
        """
        return 2 if self.dataset.temperature.size < threshold else 3

    @functools.cached_property
    def data(self):
        """Standardized data sorted by temperature, with cumulative sums."""
        index = np.argsort(self._xscaled)
        xx = self._xscaled[index]
        yy = self._yscaled[index]

        return _SegmentedData(
            xx=xx,
            yy=yy,
            sx=np.concatenate([[0], np.cumsum(xx)]),
            sy=np.concatenate([[0], np.cumsum(yy)]),
            sxx=np.concatenate([[0], np.cumsum(xx**2)]),
            syy=np.concatenate([[0], np.cumsum(yy**2)]),
            sxy=np.concatenate([[0], np.cumsum(xx * yy)]),
        )

    def _ols(self, start: int, end: int):
        """``_SegmetedData.ols``, incrementing the linear-model fit counter."""
        self._count_lm_fit += 1
        return self.data.ols(start, end)

    def optimize(self):  # ruff: ignore[too-many-locals]
        """
        Search all admissible splits.

        A split is admissible if ``xlim[0] <= T_h``,
        ``T_c - T_h > min_gap`` and ``T_c <= xlim[1]``.

        Returns
        -------
        numpy.ndarray
            Standardized change-points (T_h, T_c).
        dict
            Temperatures bracketing the selected splits
            (``sr.t_*.bracket.*``) and whether each estimated change-point
            lies between them (``sr.t_*.in_bracket``).

        Raises
        ------
        ConvergenceError
            If no admissible split is found.
        """
        m = self.min_samples()
        bound = self._xscale(self.xlim)
        min_gap = self.min_gap / self._xscaler_std
        cooling_fits = {
            j: self._ols(j, self.data.n) for j in range(2 * m, self.data.n - m + 1)
        }

        best = _BestCase()
        for i in range(m, self.data.n - 2 * m + 1):
            h = self._ols(0, i)
            if h.beta >= 0:
                continue

            for j in range(i + m, self.data.n - m + 1):
                self._count_obj_eval += 1

                c = cooling_fits[j]
                if c.beta <= 0:
                    continue

                sy_b = self.data.sy[j] - self.data.sy[i]
                syy_b = self.data.syy[j] - self.data.syy[i]
                sse_b = syy_b - (sy_b**2) / (j - i)

                sse = max(0, h.sse + sse_b + c.sse)
                if sse > best.sse:
                    continue

                baseline = sy_b / (j - i)
                th = (baseline - h.alpha) / h.beta
                tc = (baseline - c.alpha) / c.beta

                if not (bound[0] + min_gap <= th + min_gap < tc <= bound[1]):
                    continue

                best = _BestCase(
                    sse=sse,
                    th=th,
                    tc=tc,
                    heating_end=i,
                    cooling_start=j,
                )

        if np.isinf(best.sse):
            raise ConvergenceError

        temperature: np.ndarray = np.sort(self.dataset.temperature)  # not scaled
        results = {
            'sr.t_h.bracket.lower': temperature[best.heating_end - 1],
            'sr.t_h.bracket.upper': temperature[best.heating_end],
            'sr.t_c.bracket.lower': temperature[best.cooling_start - 1],
            'sr.t_c.bracket.upper': temperature[best.cooling_start],
            'sr.t_h.in_bracket': (
                self.data.xx[best.heating_end - 1]
                <= best.th
                <= self.data.xx[best.heating_end]
            ),
            'sr.t_c.in_bracket': (
                self.data.xx[best.cooling_start - 1]
                <= best.tc
                <= self.data.xx[best.cooling_start]
            ),
        }

        return np.array([best.th, best.tc]), results


def optimizer(optimizer: str, **kwargs) -> BaseOptimizer:
    """
    Create an optimizer by name.

    Parameters
    ----------
    optimizer
        Class name in this module, with or without the ``Optimizer`` suffix
        (e.g. ``'TRR'`` or ``'TRROptimizer'``).
    **kwargs
        Passed to the optimizer class.

    Returns
    -------
    BaseOptimizer
    """
    opt = optimizer if optimizer.endswith('Optimizer') else f'{optimizer}Optimizer'
    cls = getattr(sys.modules[__name__], opt)
    return cls(**kwargs)
