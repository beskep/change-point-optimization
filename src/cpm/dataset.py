"""
Synthetic datasets for the Monte Carlo evaluation of change-point optimizers.

Each synthetic dataset combines observed daily outdoor temperature with a
five-parameter (5P) change-point model

    E = E_b + beta_h * max(0, T_h - T) + beta_c * max(0, T - T_c),

where the change-points (T_h, T_c) are drawn from a bivariate normal
distribution, the baseline energy use E_b from a contaminated log-normal
distribution, and the slopes (beta_h, beta_c) are fixed percentiles of the
slopes of the measured CPM models.
"""

import dataclasses as dc
import functools
import itertools
import json
import string
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal

import more_itertools as mi
import numpy as np
import polars as pl

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


type Resolution = Literal['d', 'mo']  # temporal resolution of a dataset
type Beta = Literal['10%', '50%']  # percentile of the slope


@dc.dataclass(frozen=True)
class TemperatureSampler:
    """Provider of observed daily outdoor temperature by station and year."""

    raw: pl.LazyFrame

    @classmethod
    def create(cls, src: Path = Path('data/daily-outdoor-temperature.parquet')):
        """
        Read the preprocessed daily temperature.

        Parameters
        ----------
        src
            Parquet file with ``station``, ``date`` and ``temperature``
            columns.

        Returns
        -------
        TemperatureSampler
        """
        raw = (
            pl
            .scan_parquet(src)
            .with_columns(year=pl.col('date').dt.year())
            .with_columns(n=pl.len().over('station', 'year'))
            .filter(pl.col('n') >= 365)  # ruff:ignore[magic-value-comparison]
        )

        return cls(raw=raw)

    def cases(
        self,
        station: int | Iterable[int] | None = None,
        year: int | Iterable[int] | None = None,
    ):
        """
        List the available station-year cases.

        Parameters
        ----------
        station
            Restrict to these weather stations.
        year
            Restrict to these years.

        Returns
        -------
        polars.DataFrame
            One row per (station, year), sorted by station and year.
        """
        cases = self.raw.unique(['station', 'year'])

        if station is not None:
            s = list(mi.always_iterable(station))
            cases = cases.filter(pl.col('station').is_in(s))
        if year is not None:
            y = list(mi.always_iterable(year))
            cases = cases.filter(pl.col('year').is_in(y))

        return cases.sort('station', 'year').collect()

    def __call__(self, station: int, year: int, resolution: Resolution):
        """
        Return the temperature series of one station-year.

        Parameters
        ----------
        station
            Weather station ID.
        year
            Year.
        resolution
            Temporal resolution.

        Returns
        -------
        polars.DataFrame
        """
        raw = self.raw.filter(pl.col('station') == station, pl.col('year') == year)

        if resolution == 'd':
            return raw.collect()

        every = resolution if resolution[0] in string.digits else f'1{resolution}'
        return (
            raw
            .group_by_dynamic('date', every=every)
            .agg(pl.mean('temperature'))
            .collect()
        )


@dc.dataclass(frozen=True)
class ChangePointsGenerator:
    """
    Sampler of change-points from a bivariate normal distribution.

    Parameters
    ----------
    mean
        Mean of (T_h, T_c) [°C], shape (2,).
    cov
        Covariance matrix of (T_h, T_c), shape (2, 2).
    std
        Standard deviation of (T_h, T_c), shape (2,). Not used for sampling.
    rng
        Random number generator.

    Attributes
    ----------
    LIMITS
        Admissible range of the change-points [°C]; sampling fails if any
        sample falls outside it.
    """

    mean: np.ndarray
    cov: np.ndarray
    std: np.ndarray

    rng: np.random.Generator = dc.field(default_factory=np.random.default_rng)

    LIMITS: ClassVar[tuple[float, float]] = (0, 25)

    @classmethod
    def create(
        cls,
        src: Path = Path('data/change-points-distribution.json'),
        rng: np.random.Generator | None = None,
    ):
        """
        Read the distribution parameters estimated by ``s01distribution``.

        Parameters
        ----------
        src
            JSON file holding the change-point distribution.
        rng
            Random number generator.

        Returns
        -------
        Self
        """
        data = json.loads(src.read_text('utf-8'))
        return cls(
            mean=np.array(data['mean']),
            cov=np.array(data['cov']),
            std=np.array(data['std']),
            rng=rng or np.random.default_rng(),
        )

    def sample(self, size: int, safety_factor: float = 2):
        """
        Draw change-points satisfying T_h < T_c.

        ``size * safety_factor`` candidates are drawn, candidates with
        T_h >= T_c are discarded, and the first ``size`` are returned.

        Parameters
        ----------
        size
            Number of samples.
        safety_factor
            Oversampling factor to compensate for discarded candidates.

        Returns
        -------
        numpy.ndarray
            Samples of shape (size, 2); columns are (T_h, T_c).

        Raises
        ------
        RuntimeError
            If fewer than ``size`` valid candidates remain, or if any valid
            candidate lies outside ``LIMITS``.
        """
        samples = self.rng.multivariate_normal(
            self.mean, self.cov, size=int(size * safety_factor)
        )

        # valid samples (Th < Tc)
        samples = samples[samples[:, 0] < samples[:, 1]]

        if samples.size < size:
            msg = 'Not enough samples'
            raise RuntimeError(msg)
        if samples[(samples < self.LIMITS[0]) | (samples > self.LIMITS[1])].size:
            msg = 'Out of range'
            raise RuntimeError(msg)

        return samples[:size]

    def __call__(self):
        """
        Draw a single pair of change-points.

        Returns
        -------
        numpy.ndarray
            (T_h, T_c), shape (2,).
        """
        return self.sample(size=1, safety_factor=32).ravel()


@dc.dataclass
class _SampledParams:
    """
    Sampled inputs of one synthetic dataset, before the slopes are applied.

    Parameters
    ----------
    temperature
        Outdoor temperature [°C] at the dataset resolution.
    baseline
        Baseline energy use at the dataset resolution.
    t_h, t_c
        Reference heating and cooling change-points [°C].
    station
        Weather station ID.
    year
        Year of the data.
    resolution
        Temporal resolution.
    """

    temperature: np.ndarray
    baseline: np.ndarray
    t_h: float
    t_c: float

    station: int
    year: int
    resolution: Resolution

    def energy(self, beta_h: float, beta_c: float):
        """
        Compute the energy use of the 5P CPM.

        Parameters
        ----------
        beta_h, beta_c
            Heating and cooling slopes.

        Returns
        -------
        numpy.ndarray
            Energy use, same shape as ``temperature``.
        """
        return (
            self.baseline
            + beta_h * np.maximum(0, self.t_h - self.temperature)
            + beta_c * np.maximum(0, self.temperature - self.t_c)
        )


@dc.dataclass(frozen=True)
class Dataset:
    """
    Synthetic dataset with its reference model parameters.

    Parameters
    ----------
    temperature
        Outdoor temperature [°C] (independent variable).
    baseline
        Sampled baseline energy use of each observation.
    energy
        Energy use (dependent variable).
    expected_baseline
        Expected value of the baseline energy use distribution.
    t_h, t_c
        Reference heating and cooling change-points [°C].
    beta
        Slope percentile used for ``beta_h`` and ``beta_c``.
    beta_h, beta_c
        Reference heating and cooling slopes.
    station
        Weather station ID.
    year
        Year of the temperature data.
    resolution
        Temporal resolution.
    repeat
        Index of the Monte Carlo repetition.
    """

    temperature: np.ndarray
    baseline: np.ndarray
    energy: np.ndarray

    expected_baseline: float
    t_h: float
    t_c: float

    beta: Beta
    beta_h: float
    beta_c: float

    station: int
    year: int
    resolution: Resolution
    repeat: int

    def stats(self):
        """
        Yield the scalar fields as flat (name, value) pairs.

        Reference parameters are named ``<name>.true`` and dataset descriptors
        ``dataset.<name>``; the number of observations is yielded as
        ``dataset.count``. Array fields are skipped.

        Yields
        ------
        name : str
        value : object
        """
        for f in dc.fields(self):
            if f.type is np.ndarray:
                continue

            if f.name in {'t_h', 't_c', 'beta_h', 'beta_c'}:
                name = f'{f.name}.true'
            elif f.name in {'beta', 'station', 'year', 'resolution', 'repeat'}:
                name = f'dataset.{f.name}'
            else:
                name = f.name

            yield name, getattr(self, f.name)

        yield 'dataset.count', self.temperature.size


@dc.dataclass(frozen=True)
class BaselineParam:
    """
    Parameters of the baseline energy use distribution.

    The baseline energy use follows a contaminated log-normal distribution,
    ``E ~ (1-ϵ)LN(μ, σ²) + ϵLN(μ, λσ²)``.

    Parameters
    ----------
    mu
        Location μ of log(E).
    sigma
        Scale (standard deviation) of log(E) for the main component.
    epsilon
        Contamination ratio.
    lambda_
        Variance inflation factor of the contaminating component.
    """

    mu: float
    sigma: float
    epsilon: float
    lambda_: float

    @classmethod
    def read(cls, src: Path = Path('data/baseline-distribution.json')):
        """
        Read the parameters written by ``s01distribution baseline-params``.

        Parameters
        ----------
        src
            JSON file holding (mu, sigma, epsilon, lambda).

        Returns
        -------
        BaselineParam
        """
        data: dict[str, float] = json.loads(src.read_text('utf-8'))
        data['lambda_'] = data.pop('lambda')
        return cls(**data)

    @staticmethod
    def _expected(mu: float, var: float):
        """
        First two raw moments of a log-normal distribution.

        Parameters
        ----------
        mu
            Mean of log(X).
        var
            Variance of log(X).

        Returns
        -------
        tuple of float
            (E[X], E[X²]).
        """
        return (
            np.exp(mu + var / 2),  # E(X)
            np.exp(2 * mu + 2 * var),  # E(X^2)
        )

    @functools.cached_property
    def std(self):
        """
        Standard deviation of the baseline energy use E (not of log E).

        Returns
        -------
        float
        """
        # E(X), E(X²)
        ex1, exx1 = self._expected(self.mu, self.sigma**2)
        ex2, exx2 = self._expected(self.mu, self.lambda_ * self.sigma**2)

        ex = (1 - self.epsilon) * ex1 + self.epsilon * ex2
        exx = (1 - self.epsilon) * exx1 + self.epsilon * exx2

        # Var(X) = E(X²) - E(X)²
        return np.sqrt(exx - ex**2)


@dc.dataclass
class BetaParam:
    """
    Percentiles of the slopes of the measured CPMs.

    Parameters
    ----------
    _data
        Mapping from a statistic label (e.g. ``'10%'``) to
        (beta_h, beta_c).
    """

    _data: dict[str, tuple[float, float]]

    @classmethod
    def read(cls, src: Path = Path('data/measured-cpm-statistics.csv')):
        """
        Read the slope statistics from the measured CPM statistics.

        Parameters
        ----------
        src
            CSV file with ``statistic``, ``beta_h`` and ``beta_c`` columns.

        Returns
        -------
        BetaParam
        """
        data = (
            pl
            .read_csv(src)
            .select('statistic', 'beta_h', 'beta_c')
            .rows_by_key('statistic', unique=True)
        )
        return cls(data)

    def __call__(self, beta: Beta) -> tuple[float, float]:
        """
        Return the slopes at a percentile.

        Parameters
        ----------
        beta
            Percentile label (row of the ``statistic`` column).

        Returns
        -------
        tuple of float
            (beta_h, beta_c).
        """
        return self._data[beta]


@dc.dataclass(frozen=True)
class Generator:
    """
    Generator of synthetic datasets for one Monte Carlo condition.

    Parameters
    ----------
    repeat
        Index of the Monte Carlo repetition; also the seed of the random
        number generator.
    resolution
        Temporal resolution of the generated datasets.
    beta
        Slope percentile used for the reference slopes.
    station
        Restrict to these weather stations. All stations if None.
    year
        Restrict to these years. All years if None.
    """

    _: dc.KW_ONLY

    repeat: int

    resolution: Resolution
    beta: Beta

    station: int | Sequence[int] | None = None
    year: int | Sequence[int] | None = None

    @functools.cached_property
    def rng(self):
        """Random number generator seeded with ``repeat``."""
        return np.random.default_rng(self.repeat)

    @functools.cached_property
    def _dist(self):
        return BaselineParam.read()

    @functools.cached_property
    def _beta(self):
        return BetaParam.read()

    @functools.cached_property
    def _change_points(self):
        return ChangePointsGenerator.create(rng=self.rng)

    @functools.cached_property
    def _temperature(self):
        return TemperatureSampler.create()

    @functools.cached_property
    def expected_baseline(self):
        """
        Expected value of the baseline energy use distribution.

        Returns
        -------
        float
            ``(1-ϵ)exp(μ + σ²/2) + ϵ exp(μ + λσ²/2)``.
        """
        d = self._dist
        return (
            (1 - d.epsilon) * np.exp(d.mu + d.sigma**2 / 2)  # fmt
            + d.epsilon * np.exp(d.mu + d.lambda_ * d.sigma**2 / 2)
        )

    def sample_baseline(self, size: int):
        """
        Draw baseline energy use from the contaminated log-normal distribution.

        Parameters
        ----------
        size
            Number of samples.

        Returns
        -------
        numpy.ndarray
        """
        d = self._dist
        contaminated = self.rng.random(size) < d.epsilon
        scale = np.where(contaminated, d.sigma * np.sqrt(d.lambda_), d.sigma)
        return np.exp(self.rng.normal(loc=d.mu, scale=scale, size=size))

    def sample_params(self, station: int, year: int):
        """
        Sample the change-points and baseline energy use for one station-year.

        The baseline energy use is always sampled daily; for monthly resolution,
        temperature and baseline energy use are then averaged by month.

        Parameters
        ----------
        station
            Weather station ID.
        year
            Year.

        Returns
        -------
        _SampledParams
        """
        th, tc = self._change_points()

        temperature = self._temperature(station, year, 'd')
        baseline = self.sample_baseline(size=temperature.height)

        if self.resolution == 'd':
            t = temperature['temperature'].to_numpy()
            b = baseline
        else:
            # sampled daily, then averaged by month
            df = (
                temperature
                .with_columns(baseline=baseline)
                .group_by_dynamic('date', every='1mo')
                .agg(pl.mean('temperature', 'baseline'))
            )
            t = df['temperature'].to_numpy()
            b = df['baseline'].to_numpy()

        return _SampledParams(
            temperature=t,
            baseline=b,
            t_h=th,
            t_c=tc,
            station=station,
            year=year,
            resolution=self.resolution,
        )

    def sample_dataset(self, params: _SampledParams):
        """
        Apply the reference slopes to sampled inputs to build a dataset.

        Parameters
        ----------
        params
            Output of ``sample_params``.

        Returns
        -------
        Dataset
        """
        bh, bc = self._beta(self.beta)

        return Dataset(
            temperature=params.temperature,
            baseline=params.baseline,
            energy=params.energy(bh, bc),
            expected_baseline=self.expected_baseline,
            t_h=params.t_h,
            t_c=params.t_c,
            beta=self.beta,
            beta_h=bh,
            beta_c=bc,
            station=params.station,
            year=params.year,
            resolution=params.resolution,
            repeat=self.repeat,
        )

    def sample(self, station: int, year: int):
        """
        Generate the dataset of one station-year.

        Parameters
        ----------
        station
            Weather station ID.
        year
            Year.

        Returns
        -------
        Dataset
        """
        params = self.sample_params(station, year)
        return self.sample_dataset(params=params)

    @functools.cached_property
    def count(self):
        """Number of station-year cases generated by ``iter``."""
        return self._temperature.cases(self.station, self.year).height

    def iter(self):
        """
        Generate one dataset for every station-year case.

        Yields
        ------
        Dataset
        """
        temp = (
            self._temperature
            .cases(self.station, self.year)
            .select('station', 'year')
            .iter_rows()
        )
        for station, year in temp:
            params = self.sample_params(station=station, year=year)
            yield self.sample_dataset(params=params)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import rich
    import seaborn as sns
    import structlog
    from cyclopts import App

    from cpm.utils import equal_scale

    logger = structlog.stdlib.get_logger(__name__)
    app = App(result_action='call_if_callable')

    @app.default
    @dc.dataclass
    class _Main:
        """Visually check the sampled change-points and (baseline) energy use."""

        @staticmethod
        def _change_points():
            gen = ChangePointsGenerator.create()
            data = gen.sample(10_000)
            fig, ax = plt.subplots()
            sns.scatterplot(x=data[:, 0], y=data[:, 1], ax=ax, alpha=0.5)
            ax.margins(x=0.1, y=0.1)
            equal_scale(ax)
            ax.set_title('Change-Points')
            ax.set_xlabel('$T_h$')
            ax.set_ylabel('$T_c$')
            return fig

        @staticmethod
        def _energy():
            gen = Generator(repeat=42, resolution='d', beta='50%')
            logger.info('#sample', temperature=gen.count, count=gen.count)
            rich.print(gen.sample(station=90, year=2025))

            # temperature-energy
            _, axes = plt.subplots(2, 2, sharex=True, sharey=True)
            for ax, (beta, resolution) in zip(
                axes.ravel(),
                itertools.product(('10%', '50%'), ('d', 'mo')),
                strict=True,
            ):
                gen = Generator(repeat=42, resolution=resolution, beta=beta)
                dataset = gen.sample(station=90, year=2025)
                sns.scatterplot(
                    x=dataset.temperature, y=dataset.energy, ax=ax, alpha=0.5
                )
                ax.set_ylim(0)
                ax.set_title(f'{beta=}, {resolution=}')

            # baseline
            baseline = gen.sample_baseline(10_000_000)
            logger.info('E[X]', _value=float(gen.expected_baseline))
            logger.info('mu', _value=float(baseline.mean()), n=baseline.size)
            logger.info('sigma', _value=float(gen._dist.std))  # ruff:ignore[private-member-access]
            logger.info('s', _value=float(baseline.std()), n=baseline.size)

            _, ax = plt.subplots()
            baseline = gen.sample_baseline(10000)
            sns.histplot(baseline, kde=True, log_scale=True, ax=ax)
            ax.axvline(gen.expected_baseline)
            ax.set_title('Baseline Distribution')

        def __call__(self):
            self._change_points()
            self._energy()
            plt.show()

    app()
