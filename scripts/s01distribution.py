"""
Analyse the distributions of the raw CPM models estimated by brute force.

The resulting distributions of the change-points, slopes and baseline energy use
are the inputs of the Monte Carlo simulation (``cpm.dataset``).
"""

import dataclasses as dc
import functools
import json
from collections.abc import Sequence  # ruff:ignore[typing-only-standard-library-import]
from pathlib import Path
from typing import ClassVar, Literal

import cyclopts
import matplotlib.patheffects as pe
import more_itertools as mi
import numpy as np
import polars as pl
import scipy.optimize as opt
import seaborn as sns
import structlog
from cyclopts.config import Toml
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from rich.progress import track
from scipy import stats

from cpm import dataset
from cpm.dirs import Dirs  # ruff:ignore[typing-only-first-party-import]
from cpm.utils import App, equal_scale

EPSILON = 1e-15
"""Offset added to the PDF before taking the logarithm."""


app = App(
    config=[
        Toml('env.toml', use_commands_as_keys=False),
        Toml('conf.toml', root_keys='distribution', use_commands_as_keys=False),
    ]
)
logger = structlog.stdlib.get_logger()


@dc.dataclass(frozen=True)
@cyclopts.Parameter(name='*')
class Config:
    """
    Common configuration of the distribution commands.

    Parameters
    ----------
    dirs
        Directory layout.
    min_baseline_samples
        Minimum number of baseline-state days for a building to be included
        in the baseline energy use distribution fitting.
    min_r2
        Minimum R² of the measured CPM models.
    workers
        Number of parallel workers for distribution fitting.
    """

    dirs: Dirs

    min_baseline_samples: int = 30
    min_r2: float | None = 0.5
    workers: int = 1


@app.command
@dc.dataclass
class RawDist:
    """
    Analyse the change-point, slope and R² distributions of the measured models.

    Parameters
    ----------
    percentiles
        Percentiles reported in the descriptive statistics.
    """

    conf: Config
    percentiles: tuple[float, ...] = (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)

    @functools.cached_property
    def _output(self):
        """Output directory (``dirs.dist.raw``), created if missing."""
        p = self.conf.dirs.dist.raw
        p.mkdir(parents=True, exist_ok=True)
        return p

    def __call__(self):
        """
        Describe the model parameters and fit the change-point distribution.

        Only models with R² >= ``min_r2`` are used. Writes the descriptive
        statistics of (E_b, beta_h, beta_c, T_h, T_c) to
        ``data/measured-cpm-statistics.csv`` and the mean, covariance,
        standard deviation and correlation of (T_h, T_c), used as a bivariate
        normal distribution, to ``data/change-points-distribution.json``.
        """
        data = (
            pl
            .scan_parquet(self.conf.dirs.data / '01.models-filtered.parquet')
            .filter(pl.col('r2') >= self.conf.min_r2)
            .collect()
        )

        # beta
        (
            data
            .filter(pl.col('names').is_in(['HDD', 'CDD']))
            .select('coef')
            .describe(percentiles=self.percentiles)
            .write_csv(self._output / '01.measured-cpm-beta.csv')
        )

        wide = (
            (data)
            .with_columns(
                pl.col('names').replace_strict({
                    'Intercept': 'baseline',
                    'HDD': 'beta_h',
                    'CDD': 'beta_c',
                })
            )
            .pivot('names', index='id', values='coef')
        )

        change_points = (
            data
            .drop_nulls('change_points')
            .with_columns(pl.col('names').replace_strict({'CDD': 'T_c', 'HDD': 'T_h'}))
            .pivot('names', index='id', values='change_points')
        )

        # base, betah, betac, th, tc
        params = wide.join(change_points, on='id').drop('id')
        desc = (
            params
            .describe(percentiles=self.percentiles, interpolation='linear')
            .filter(pl.col('statistic').is_in(['min', 'max']).not_())
            .with_columns()
        )
        desc.write_parquet(self._output / '01.measured-cpm-statistics.parquet')
        desc.write_csv(self._output / '01.measured-cpm-statistics.csv')
        desc.write_csv('data/measured-cpm-statistics.csv')

        _ = (
            params
            .with_columns(pl.col('beta_h', 'beta_c') * 1000)  # kWh/m²K -> Wh/m²K
            .describe(percentiles=self.percentiles, interpolation='linear')
            .write_csv(self._output / '01.measured-cpm-statistics.Wh_m²K.csv')
        )

        # multivariate normal distribution of the change-points
        cp = change_points.select('T_h', 'T_c').to_numpy()
        dist: dict = {
            'mean': cp.mean(axis=0),
            'cov': np.cov(cp.T, ddof=1),
            'std': np.std(cp, axis=0, ddof=1),
            'r': np.corrcoef(cp.T),
        }
        Path('data/change-points-distribution.json').write_text(
            json.dumps({k: v.tolist() for k, v in dist.items()}),
            encoding='utf-8',
        )

        self._plot_params(data)
        self._plot_cp(change_points, dist)

    def _plot_params(self, data: pl.DataFrame):
        """
        Plot histograms of R² and of the heating/cooling slopes.

        Parameters
        ----------
        data
            Measured CPM models in long format.
        """
        # r2
        fig = Figure()
        sns.histplot(data['r2'], kde=True, ax=fig.subplots())
        fig.savefig(self._output / '02.r2.svg')

        # beta
        fig = Figure()
        ax = fig.subplots()
        sns.histplot(
            data=data.filter(pl.col('names').is_in(['HDD', 'CDD'])).with_columns(
                pl.col('names').replace_strict({'HDD': 'Heating', 'CDD': 'Cooling'})
            ),
            x='coef',
            hue='names',
            hue_order=['Cooling', 'Heating'],
            kde=True,
            ax=ax,
            log_scale=True,
        )
        if legend := ax.get_legend():
            legend.set_title('')
        ax.set_xlabel('Coefficient [kWh/m²°C]')
        fig.savefig(self._output / '02.beta.svg')

    def _plot_cp(self, data: pl.DataFrame, dist: dict[str, np.ndarray]):
        """
        Plot the scatter of (T_h, T_c) with their correlation coefficient.

        Parameters
        ----------
        data
            Change-points in wide format (``T_h``, ``T_c``).
        dist
            Distribution parameters; ``dist['r']`` is the correlation matrix.
        """
        fig = Figure((7, 6.5, 'cm'))
        ax = fig.subplots()

        sns.scatterplot(
            data=data, x='T_h', y='T_c', ax=ax, s=16, c='slategray', alpha=0.4
        )
        equal_scale(ax, locator=MaxNLocator('auto', steps=[1, 2, 5, 10]))

        ax.set_xlabel('Heating Change-Point [°C]', labelpad=-1)
        ax.set_ylabel('Cooling Change-Point [°C]')

        r = dist['r'][0][1]
        ax.annotate(
            f'$r={r:.3f}$',
            (20, 5),
            xytext=(-2, 4),
            textcoords='offset points',
            ha='right',
            path_effects=[pe.Stroke(linewidth=4, foreground='w'), pe.Normal()],
        )

        fig.savefig(self._output / '03.change-points-dist.svg')
        fig.savefig(self._output / '03.change-points-dist.pdf')


@dc.dataclass
class _DistributionFitter:
    """
    Base class of the maximum-likelihood fitters of contaminated distributions.

    A contaminated distribution is a two-component mixture
    ``(1-ϵ)F(θ) + ϵF(θ')``, where the second component has an inflated scale
    controlled by λ.

    Parameters
    ----------
    data
        Samples (daily baseline EUI of one building).
    max_epsilon
        Upper bound of the contamination ratio ϵ.
    workers
        Number of parallel workers.

    Attributes
    ----------
    PARAMS
        Names of the distribution parameters; the last two are always
        ``epsilon`` and ``lambda``.
    """

    data: np.ndarray
    max_epsilon: float = 0.5  # maximum ratio of outliers
    workers: int = 1

    PARAMS: ClassVar[tuple[str, ...]]

    def objective(self, params: tuple[float, ...], data: np.ndarray) -> float:
        """
        Negative log-likelihood.

        Parameters
        ----------
        params
            Distribution parameters in the order of ``PARAMS``.
        data
            Samples.

        Returns
        -------
        float
        """
        raise NotImplementedError

    @functools.cached_property
    def bound(self) -> list[tuple[float, ...]]:
        """Bounds of the parameters, in the order of ``PARAMS``."""
        raise NotImplementedError

    def fit(self) -> opt.OptimizeResult:
        """
        Minimize the negative log-likelihood by differential evolution.

        Returns
        -------
        scipy.optimize.OptimizeResult
            Optimization result.
        """
        return opt.differential_evolution(
            self.objective,
            bounds=self.bound,
            args=(self.data,),
            strategy='rand1bin',
            popsize=25,
            tol=0.001,
            polish=True,
            rng=42,
            workers=self.workers,
        )

    @staticmethod
    def _negative_log_likelihood(pdf: np.ndarray):
        """
        Negative log-likelihood from PDF values.

        Parameters
        ----------
        pdf
            PDF evaluated at the samples.

        Returns
        -------
        float
        """
        return -np.log(pdf + EPSILON).sum()

    def __call__(self):
        """
        Fit the distribution and summarise the result.

        The fit is regarded as converged if the optimizer succeeded and ϵ and
        λ are not at their upper/lower bounds.

        Returns
        -------
        dict
            Fitter name, parameters, negative log-likelihood, sample size,
            AIC, BIC and success/convergence flags.
        """
        r = self.fit()
        params = dict(zip(self.PARAMS, r.x, strict=True))
        k = r.x.size
        nll = r.fun
        converged = bool(
            r.success
            and params['epsilon'] < self.bound[2][1]
            and params['lambda'] > self.bound[3][0]
            and params['lambda'] < self.bound[3][1]
        )

        return {
            'distribution': self.__class__.__name__,
            **params,
            'nll': nll,
            'n': self.data.size,
            'aic': 2 * k + 2 * nll,
            'bic': k * np.log(self.data.size) + 2 * nll,
            'success': r.success,
            'converged': converged,
        }


@dc.dataclass
class _GaussianMixtureFitter(_DistributionFitter):
    """Contaminated normal distribution, ``(1-ϵ)N(μ, σ²) + ϵN(μ, λσ²)``."""

    PARAMS: ClassVar[tuple[str, ...]] = ('mu', 'sigma', 'epsilon', 'lambda')

    @functools.cached_property
    def bound(self):
        return [
            (-20, 20),
            (1e-6, 20),
            (0.0, self.max_epsilon),  # epsilon <= max_epsilon
            (1.0 + 1e-6, 100.0),  # 1 < lambda
        ]

    def objective(self, params: tuple[float, ...], data: np.ndarray):
        mu, sigma, epsilon, lambda_ = params
        pdf = (
            (1 - epsilon) * stats.norm.pdf(data, mu, sigma)  # fmt
            + epsilon * stats.norm.pdf(data, mu, np.sqrt(lambda_) * sigma)
        )

        return self._negative_log_likelihood(pdf=pdf)


@dc.dataclass
class _LogGaussianMixtureFitter(_GaussianMixtureFitter):
    """
    Contaminated log-normal distribution, ``log(E) ~ (1-ϵ)N(μ, σ²) + ϵN(μ, λσ²)``.

    The likelihood is evaluated for E, i.e. including the Jacobian term
    ``Σ log(E)``.
    """

    def objective(self, params, data):
        mu, sigma, epsilon, lambda_ = params

        logdata = np.log(data)
        pdf1 = stats.norm.pdf(logdata, mu, sigma)
        pdf2 = stats.norm.pdf(logdata, mu, np.sqrt(lambda_) * sigma)
        pdf = (1 - epsilon) * pdf1 + epsilon * pdf2
        jacobian_term = logdata.sum()

        return self._negative_log_likelihood(pdf=pdf) + jacobian_term


@dc.dataclass
class _WeibullMixtureFitter(_DistributionFitter):
    """Contaminated Weibull distribution, ``(1-ϵ)Weibull(c, s) + ϵWeibull(c, √λ·s)``."""

    PARAMS: ClassVar[tuple[str, ...]] = ('c', 'scale', 'epsilon', 'lambda')

    @functools.cached_property
    def bound(self):
        return [
            (1e-3, 100),
            (1e-3, 10),
            (0, self.max_epsilon),
            (1.0 + 1e-6, 1000.0),
        ]

    def objective(self, params, data):
        c, scale, epsilon, lambda_ = params

        pdf1 = stats.weibull_min.pdf(data, c=c, scale=scale)
        pdf2 = stats.weibull_min.pdf(data, c=c, scale=np.sqrt(lambda_) * scale)
        pdf = (1 - epsilon) * pdf1 + epsilon * pdf2

        return self._negative_log_likelihood(pdf=pdf)


@dc.dataclass
class _Chi2MixtureFitter(_DistributionFitter):
    """
    Contaminated scaled chi-squared distribution.

    ``(1-ϵ)χ²(df, s) + ϵχ²(df, λs)``.
    """

    PARAMS: ClassVar[tuple[str, ...]] = ('df', 'scale', 'epsilon', 'lambda')

    @functools.cached_property
    def bound(self):
        return [
            (0.1, 500.0),
            (1e-3, 50.0),
            (0.0, self.max_epsilon),
            (1.0 + 1e-6, 100.0),
        ]

    def objective(self, params, data):
        df, scale, epsilon, lambda_ = params
        pdf = (
            (1 - epsilon) * stats.chi2.pdf(data, df=df, scale=scale)  # fmt
            + epsilon * stats.chi2.pdf(data, df=df, scale=lambda_ * scale)
        )

        return self._negative_log_likelihood(pdf)


@dc.dataclass
class _DistPaths:
    """
    Output directories of the baseline energy use distribution analysis.

    Parameters
    ----------
    root
        ``dirs.dist.baseline``.
    raw
        Histograms of the raw data.
    dist
        Fitted distribution parameters of each building.
    params
        Summaries of the fitted distributions.
    """

    root: Path
    raw: Path
    dist: Path
    params: Path

    @classmethod
    def create(cls, conf: Config):
        """
        Create the directories.

        Parameters
        ----------
        conf
            Configuration.

        Returns
        -------
        _DistPaths
        """
        root = conf.dirs.dist.baseline
        paths = cls(
            root=root,
            raw=root / '00.raw',
            dist=root / '01.dist',
            params=root / '02.params',
        )

        for p in dc.asdict(paths).values():
            p.mkdir(exist_ok=True)

        return paths


@app.command
@dc.dataclass
class FitBaselineDistribution:
    """
    Fit baseline energy use distributions for the Monte Carlo simulation.

    Parameters
    ----------
    hist
        Plot the histogram of the baseline energy use of each building.
    fit
        Fit the candidate distributions.
    """

    _: dc.KW_ONLY

    hist: bool = False
    fit: bool = True
    conf: Config

    @functools.cached_property
    def _paths(self):
        """Output directories."""
        return _DistPaths.create(self.conf)

    @functools.cached_property
    def data(self):
        """
        Daily baseline-state EUI on working days.

        Returns
        -------
        polars.LazyFrame
        """
        return (
            pl
            .scan_parquet(self.conf.dirs.data / '04.data.parquet')
            .filter(
                pl.col('is_holiday').not_(),
                pl.col('state') == 'baseline',
                pl.col('eui') > 0,
            )
            .drop_nulls('eui')
        )

    @functools.cached_property
    def ids(self):
        """IDs of the buildings with at least ``min_baseline_samples`` days."""
        return (
            self.data
            .group_by('id')
            .len()
            .filter(pl.col('len') >= self.conf.min_baseline_samples)
            .collect()
            .sort('id')
            .to_series(0)
            .to_list()
        )

    def _hist(self):
        """Plot the baseline energy use histogram of each building."""
        for idx, id_ in enumerate(track(self.ids)):
            data = self.data.filter(pl.col('id') == id_).collect()['eui'].to_numpy()
            fig = Figure()
            ax = fig.subplots()
            sns.histplot(data, kde=True, ax=ax)
            ax.text(0.02, 0.98, f'n={data.size}', transform=ax.transAxes, va='top')
            fig.savefig(self._paths.raw / f'{idx:03d}.{id_}.png')

    def _fit(
        self,
        dist: Literal['gaussian', 'log-gaussian', 'chi2', 'weibull'],
        max_epsilon: float = 0.5,
    ):
        """
        Fit a contaminated distribution to the baseline energy use of each building.

        The candidate distributions are::

            E      = (1-ϵ)N(μ,σ²)      + ϵN(μ,λσ²)
            log(E) = (1-ϵ)N(μ,σ²)      + ϵN(μ,λσ²)
            E      = (1-ϵ)χ²(df,s)     + ϵχ²(df,λs)
            E      = (1-ϵ)Weibull(c,s) + ϵWeibull(c,√λ·s)

        Parameters
        ----------
        dist
            Distribution.
        max_epsilon
            Upper bound of the contamination ratio ϵ.

        Raises
        ------
        ValueError
            If ``dist`` is unknown.
        """
        match dist:
            case 'gaussian':
                fitter = _GaussianMixtureFitter
            case 'log-gaussian':
                fitter = _LogGaussianMixtureFitter
            case 'chi2':
                fitter = _Chi2MixtureFitter
            case 'weibull':
                fitter = _WeibullMixtureFitter
            case _:
                raise ValueError(dist)

        models = []
        kwargs = {
            'max_epsilon': max_epsilon,
            'workers': self.conf.workers,
        }
        for id_ in track(self.ids):
            data = (
                self.data
                .filter(pl.col('id') == id_)
                .select('eui')
                .collect()
                .to_numpy()
                .ravel()
            )

            r = fitter(data, **kwargs)()
            models.append({'id': id_, **r})

        params = pl.from_dicts(models).with_columns(contaminated=pl.col('epsilon') != 0)

        name = f'{dist} {max_epsilon=}'.replace("'", '')
        path = self._paths.dist / f'{name}.parquet'

        params.write_parquet(path)
        params.write_csv(path.with_suffix('.csv'))

    def __call__(self):  # ruff: ignore[undocumented-public-method]
        logger.info('dataset', n=len(self.ids))

        skew = (
            self.data
            .filter(pl.col('id').is_in(self.ids))
            .group_by('id')
            .agg(pl.col('eui').skew().alias('skew'))
            .drop_nulls('skew')
            .drop_nans('skew')
            .select(
                pl.len(),
                pl.mean('skew').alias('mean'),
                pl.std('skew').alias('std'),
                pl.median('skew').alias('median'),
                (pl.col('skew') > 0).sum().alias('pos-count'),
                (pl.col('skew') > 0).mean().alias('pos-ratio'),
            )
            .collect()
        )
        skew.write_csv(self._paths.dist / 'baseline.skew.csv')
        logger.info('baseline skew', **skew.to_dicts()[0])

        if self.hist:
            self._hist()

        if not self.fit:
            return

        for dist in ('gaussian', 'log-gaussian', 'weibull', 'chi2'):
            logger.info('fitting', distribution=dist)
            self._fit(dist)


@app.command
@dc.dataclass
class BaselineDist:
    """Compare the fitted baseline energy use distributions."""

    conf: Config

    @functools.cached_property
    def _paths(self):
        """Output directories."""
        return _DistPaths.create(self.conf)

    @functools.cached_property
    def sources(self):
        """Parquet files of the fitted distributions."""
        return list(self._paths.dist.glob('*.parquet'))

    @functools.cached_property
    def data(self):
        """
        Converged fits of all distributions.

        Parameter names are unified as ``shape`` and ``scale``.

        Raises
        ------
        ValueError
            If the fits were obtained with more than one method.
        """

        def it():
            for src in self.sources:
                yield (
                    pl
                    .scan_parquet(src)
                    .rename(
                        {'mu': 'shape', 'c': 'shape', 'df': 'shape', 'sigma': 'scale'},
                        strict=False,
                    )
                    .with_columns(
                        pl
                        .col('distribution')
                        .str.strip_prefix('_')
                        .str.strip_suffix('Fitter')
                        .str.strip_suffix('Mixture')
                    )
                )

        data = (
            pl
            .concat(it(), how='diagonal_relaxed')
            .filter(pl.col('converged'))
            .collect()
        )
        if data['method'].n_unique() != 1:
            raise ValueError

        return data

    def desc(self, *, contaminated_only: bool = False):
        """
        Write descriptive statistics and BIC comparison of the distributions.

        For each building, the distribution with the lowest BIC is counted as
        preferred.

        Parameters
        ----------
        contaminated_only
            Use only fits with ϵ != 0.
        """
        root = self._paths.params
        dist = 'distribution'

        data = self.data.with_columns(
            pl.col(dist).replace_strict({
                'Gaussian': '01.Normal',
                'LogGaussian': '02.LogNormal',
                'Chi2': '03.ChiSquared',
                'Weibull': '04.Weibull',
            })
        ).sort(dist)

        if contaminated_only:
            data = data.filter(pl.col('contaminated'))
            prefix = '(ContaminatedOnly)'
        else:
            prefix = ''

        data = data.with_columns(
            (pl.col('bic').rank().over('id') == 1).alias('preferred')
        )

        def describe():
            for (d,), df in data.drop('aic', 'bic').group_by(dist, maintain_order=True):
                yield df.drop(dist).describe().select(pl.lit(d).alias(dist), pl.all())

        desc = pl.concat(describe())
        desc.write_csv(root / f'{prefix}01.describe.csv')

        (
            data
            .pivot(dist, index='id', values='bic')
            .drop_nulls()
            .unpivot(index='id', variable_name=dist, value_name='bic')
            .group_by(dist)
            .agg(pl.mean('bic').alias('bic.mean'), pl.median('bic').alias('bic.median'))
            .join(
                data.group_by(dist).agg(pl.sum('preferred')),
                on=dist,
                how='left',
                validate='1:1',
            )
            .sort(dist)
            .write_csv(root / f'{prefix}02.bic.csv')
        )

    def plot_bic(self):
        """Plot the BIC of each distribution."""
        data = (
            self.data
            .pivot('distribution', index='id', values='bic')
            .drop_nulls()
            .unpivot(index='id', variable_name='distribution', value_name='bic')
        )
        fig = Figure()
        ax = fig.subplots()
        sns.barplot(data, x='bic', y='distribution', ax=ax)
        fig.savefig(self._paths.root / '03.bic.png')

    def plot_epsilon_bic(self):
        """Plot the contamination ratio ϵ against BIC."""
        fig = Figure()
        ax = fig.subplots()
        sns.scatterplot(
            self.data, x='epsilon', y='bic', hue='distribution', ax=ax, alpha=0.5
        )
        fig.savefig(self._paths.root / '03.epsilon vs bic.png')

    def __call__(self):  # ruff: ignore[undocumented-public-method]
        self.data.write_csv(self._paths.params / '00.raw.csv')

        self.desc()
        self.desc(contaminated_only=True)

        self.plot_bic()
        self.plot_epsilon_bic()


@app.command
def baseline_params(*, conf: Config):
    """
    Extract the median baseline energy use parameters for the simulation.

    The medians of the converged log-normal fits are written to
    ``data/baseline-distribution.json``.

    Returns
    -------
    dict[str, float]
        Median of each parameter (mu, sigma, epsilon, lambda).
    """
    paths = _DistPaths.create(conf)
    src = mi.one(paths.dist.glob('log-gaussian*.parquet'))

    variables = ['mu', 'sigma', 'epsilon', 'lambda']
    array = (
        pl
        .scan_parquet(src)
        .filter(pl.col('converged'))
        .select(pl.median(*variables))
        .collect()
        .to_numpy()
        .ravel()
    )

    params = dict(zip(variables, array, strict=True))
    Path('data/baseline-distribution.json').write_text(
        json.dumps(params), encoding='utf-8'
    )

    return params


@app.command
@dc.dataclass
class PowerAnalysis:
    """
    Analyse the statistical power of the slope t-tests.

    Parameters
    ----------
    change_points
        Change-points (T_h, T_c) [°C] used to build the HDD/CDD regressors.
    station
        ASOS station ID of the temperature data.
    year
        Year of the temperature data.
    alpha
        Significance level of the one-sided t-test.
    """

    _: dc.KW_ONLY

    change_points: tuple[float, float] = (11.0, 16.1)
    station: int = 108  # Seoul
    year: int = 2025
    resolution: Sequence[dataset.Resolution] = ('d', 'mo')

    alpha: float = 0.05
    conf: Config

    @functools.cached_property
    def sigma(self):
        """
        Standard deviation of the baseline energy use.

        Used as the residual standard deviation of the dependent variable.
        """
        return dataset.BaselineParam.read().std

    @functools.cached_property
    def beta_set(self):
        """Slopes (``betah``, ``betac``) of measured models with R² >= ``min_r2``."""
        # beta distribution of the CPM models fitted to measured data
        return (
            pl
            .scan_parquet(self.conf.dirs.data / '01.models-filtered.parquet')
            .filter(
                pl.col('r2') >= self.conf.min_r2, pl.col('names').is_in(['HDD', 'CDD'])
            )
            .with_columns(pl.col('names').replace({'HDD': 'betah', 'CDD': 'betac'}))
            .collect()
            .pivot('names', index='id', values='coef')
            .select('id', 'betah', 'betac')
        )

    @functools.cached_property
    def temperature(self) -> dict[dataset.Resolution, pl.DataFrame]:
        """Design matrix (intercept, HDD, CDD) of each resolution."""
        return {
            r: dataset.TemperatureSampler.create()(
                station=self.station, year=self.year, resolution=r
            ).select(
                ones=pl.lit(1, dtype=pl.Float64),
                HDD=pl.max_horizontal(0, self.change_points[0] - pl.col('temperature')),
                CDD=pl.max_horizontal(0, pl.col('temperature') - self.change_points[1]),
            )
            for r in self.resolution
        }

    def _power_helper(
        self,
        resolution: dataset.Resolution,
        percentile: float,
        beta_true: tuple[float, float],
    ):
        """
        Power for fixed change-points from the OLS standard errors.

        The noncentrality parameter is ``beta / (sigma * sqrt(C_jj))``, where ``C_jj``
        is a diagonal element of ``(XᵀX)⁻¹``.

        Parameters
        ----------
        resolution
            Temporal resolution.
        percentile
            Percentile of the reference slopes.
        beta_true
            Reference slopes (beta_h, beta_c).

        Returns
        -------
        list of dict
            One record per slope.
        """
        temperature = self.temperature[resolution]

        n = temperature.height
        dof = n - 3  # k=2 independent variables, dof=n-k-1

        x = temperature.select('ones', 'HDD', 'CDD').to_numpy()

        # Cjj=[C00, C11, C22], diagonal elements of inv(X'X)
        cjj = np.diag(np.linalg.inv(x.T @ x))
        lambdas = np.array(beta_true) / (self.sigma * np.sqrt(cjj[1:]))

        crit = stats.t.ppf(1 - self.alpha, df=dof)
        power = 1 - stats.nct.cdf(crit, df=dof, nc=lambdas)

        variables = ('beta_h', 'beta_c')
        d = {'resolution': resolution, 'n': n, 'percentile': percentile}
        return [
            {**d, 'variable': v, 'beta_true': t, 'lambda': lm, 'power': p}
            for v, t, lm, p in zip(variables, beta_true, lambdas, power, strict=True)
        ]

    def power_fixed_change_points(self):
        """
        Power for fixed change-points at several slope percentiles.

        Returns
        -------
        polars.DataFrame
        """
        percentiles = (0.05, 0.1, 0.2, 0.5)
        beta_percentile = (
            self.beta_set
            .describe(percentiles=percentiles, interpolation='linear')
            .select('statistic', 'betah', 'betac')
            .with_columns()
        )

        results = []
        for p in percentiles:
            row = beta_percentile.row(
                by_predicate=pl.col('statistic') == f'{p:.0%}',
                named=True,
            )
            for resolution in self.resolution:
                results.extend(
                    self._power_helper(
                        resolution=resolution,
                        percentile=p,
                        beta_true=(row['betah'], row['betac']),
                    )
                )

        return pl.from_dicts(results).sort(pl.all())

    def __call__(self):  # ruff: ignore[undocumented-public-method]
        output = self.conf.dirs.dist.root

        power = self.power_fixed_change_points()
        power.write_csv(output / '03.statistical-power.csv', include_bom=True)

        return power


if __name__ == '__main__':
    app()
