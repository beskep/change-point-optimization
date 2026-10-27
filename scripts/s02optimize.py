"""Run the change-point optimizers on synthetic datasets (Monte Carlo experiment)."""

import dataclasses as dc
import functools
import itertools
import warnings
from collections.abc import Sequence  # ruff: ignore[typing-only-standard-library-import]

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import rich
import seaborn as sns
import structlog
from cyclopts.config import Toml
from matplotlib.figure import Figure
from statsmodels.tools.sm_exceptions import SingularMatrixWarning
from tqdm import TqdmExperimentalWarning
from tqdm.rich import tqdm

import cpm.dataset as cd
import cpm.optimizer as co
from cpm.dataset import Beta, Resolution  # ruff:ignore[typing-only-first-party-import]
from cpm.dirs import Dirs  # ruff:ignore[typing-only-first-party-import]
from cpm.utils import App

logger = structlog.stdlib.get_logger()
app = App(
    config=[
        Toml('env.toml', allow_unknown=True, use_commands_as_keys=False),
        Toml('conf.toml', root_keys='optimize'),
    ],
    result_action=['call_if_callable', 'print_non_int_sys_exit'],
    exit_on_error=False,
)


@app.command
@dc.dataclass
class Example:
    """
    Plot example CPM models fitted to measured data.

    Parameters
    ----------
    institution
        Institution ID.
    station
        ASOS station ID of the building.
    """

    institution: str = 'DB_B7AE8782-B65B-8EED-E050-007F01001D51'
    station: int = 108

    _: dc.KW_ONLY
    dirs: Dirs

    @functools.cached_property
    def ami(self):
        """Daily AMI energy use."""
        return pl.scan_parquet(self.dirs.data / '02.energy.parquet')

    @functools.cached_property
    def weather(self):
        """Daily outdoor temperature."""
        return pl.scan_parquet('data/daily-outdoor-temperature.parquet')

    def plot(self):
        """Fit a TRR model to the working-day EUI of a building and plot it."""
        ami = (
            self.ami
            .filter(pl.col('id') == self.institution, pl.col('is_holiday').not_())
            .collect()
            .with_columns()
        )
        temp = (
            self.weather
            .filter(pl.col('station') == self.station)
            .select('date', 'temperature')
            .collect()
        )
        data = ami.join(temp, on='date')

        t = data['temperature'].to_numpy()
        dataset = cd.Dataset(
            temperature=t,
            baseline=np.ones_like(t),
            energy=data['eui'].to_numpy(),
            expected_baseline=0,
            t_h=0,
            t_c=0,
            beta='50%',
            beta_h=0,
            beta_c=0,
            station=self.station,
            year=0,
            resolution='d',
            repeat=0,
        )

        optimizer = co.TRROptimizer(dataset)
        model = optimizer()

        xx = np.array([t.min(), model['t_h.pred'], model['t_c.pred'], t.max()])
        yy = (
            model['baseline.pred']
            + model['beta_h.pred'] * np.maximum(0, model['t_h.pred'] - xx)
            + model['beta_c.pred'] * np.maximum(0, xx - model['t_c.pred'])
        )

        fig = Figure((8 * 1.2, 6 * 1.2, 'cm'))
        ax = fig.subplots()

        sns.scatterplot(
            data, x='temperature', y='eui', ax=ax, color='slategray', alpha=0.5, s=12
        )
        ax.plot(xx, yy, '-', color='darkslategray', alpha=0.75)

        ls: dict = {'ls': '--', 'lw': 1, 'color': 'darkslategray', 'alpha': 0.5}
        ax.axvline(model['t_h.pred'], **ls)
        ax.axvline(model['t_c.pred'], **ls)
        ax.axhline(model['baseline.pred'], **ls)

        ax.set_ylim(0)
        ax.set_xlabel('Outdoor Air Temperature [°C]', labelpad=-1)
        ax.set_ylabel('Energy Use Intensity [kWh/m²]')

        trsf = ax.transAxes + ax.transData.inverted()
        corner = trsf.transform([0, 0])
        gap = trsf.transform([0.01, 0.01]) - corner

        ax.text(model['t_h.pred'] - gap[0], corner[1] + 5 * gap[1], '$T_h$', ha='right')
        ax.text(model['t_c.pred'] + 1.5 * gap[0], corner[1] + 5 * gap[1], '$T_c$')
        ax.text(
            corner[0] + 3 * gap[0],
            model['baseline.pred'] - 0.8 * gap[1],
            '$E_b$',
            va='top',
        )

        return fig

    def __call__(self):  # ruff: ignore[undocumented-public-method]
        with plt.rc_context({'axes.grid': False}):
            fig = self.plot()

        fig.savefig(self.dirs.dist.root / 'CPM.example.svg')
        fig.savefig(self.dirs.dist.root / 'CPM.example.pdf')


@dc.dataclass(frozen=True)
class _OptimizerConfig:
    """
    Optimizer specification read from the configuration file.

    Parameters
    ----------
    optimizer
        Optimizer name (see ``cpm.optimizer.optimizer``).
    linear_model
        Linear model.
    object
        Objective function.
    steps
        Comma-separated grid steps of ``IMTOptimizer`` (e.g. ``'2.5,0.25'``).
    step
        Grid step of ``GridSearchOptimizer``.
    loss
        Loss function of the TRR optimizers.

    Notes
    -----
    Unset (None) options are not passed, so the optimizer defaults apply.
    """

    optimizer: str
    linear_model: co.LinearModel | None = None
    object: co.ObjectFunction | None = None
    steps: str | None = None  # NOTE: cyclopts cannot parse list[float]
    step: float | None = None
    loss: str | None = None

    def __call__(self, dataset: cd.Dataset):
        """
        Create the optimizer for a dataset.

        Parameters
        ----------
        dataset
            Dataset to fit.

        Returns
        -------
        cpm.optimizer.BaseOptimizer
        """
        kwargs = {k: v for k, v in dc.asdict(self).items() if v is not None}
        if 'steps' in kwargs:
            kwargs['steps'] = tuple(float(x) for x in kwargs['steps'].split(','))

        return co.optimizer(**kwargs, dataset=dataset)

    def __str__(self):
        """Return the set options joined by ``-``, used in output file names."""
        return '-'.join(str(x) for x in dc.asdict(self).values() if x is not None)


@app.command
def test_run():
    """Run several optimizers on one synthetic dataset and plot the fits."""
    dataset = cd.Generator(repeat=42, resolution='d', beta='50%').sample(
        station=108, year=2024
    )

    optimizers = [
        _OptimizerConfig('IMT'),
        _OptimizerConfig('GridSearch'),
        _OptimizerConfig('Muggeo'),
        _OptimizerConfig('TRR'),
        _OptimizerConfig('MultiStartTRR'),
        _OptimizerConfig('SegmentedRegression'),
    ]
    models = [x(dataset)() for x in optimizers]
    rich.print(models)
    models = [x for x in models if x.get('converged')]

    fig, axes = plt.subplots(ncols=(len(models) + 1) // 2, nrows=2)
    for model, ax in zip(models, axes.ravel(), strict=False):
        sns.scatterplot(x=dataset.temperature, y=dataset.energy, ax=ax, alpha=0.5)
        xx = np.array([
            dataset.temperature.min(),
            model['t_h.pred'],
            model['t_c.pred'],
            dataset.temperature.max(),
        ])
        ax.plot(
            xx,
            model['baseline.pred']
            + model['beta_h.pred'] * np.maximum(0, model['t_h.pred'] - xx)
            + model['beta_c.pred'] * np.maximum(0, xx - model['t_c.pred']),
        )
        ax.axvline(dataset.t_h, color='gray', linestyle='--')
        ax.axvline(dataset.t_c, color='gray', linestyle='--')
        ax.update_datalim([[0, 0]], updatex=False)
        ax.autoscale_view()
        ax.set_ylim(0)
        ax.set_title(f'{model["optimizer"]}')

    plt.show()
    plt.close(fig)


@app.command
@dc.dataclass(frozen=True)
class Optimize:
    """
    Run every optimizer on the synthetic datasets of all conditions.

    Parameters
    ----------
    repeat
        Number of Monte Carlo repetitions.
    resolution
        Temporal resolutions of the datasets.
    beta
        Slope percentiles of the datasets.
    reverse
        Process the cases in reverse order.
    """

    _: dc.KW_ONLY
    dirs: Dirs
    optimizers: Sequence[_OptimizerConfig]

    repeat: int = 20
    resolution: Sequence[Resolution] = ('d', 'mo')
    beta: Sequence[Beta] = ('10%', '50%')
    reverse: bool = False

    def _optimize(
        self,
        optimizer: _OptimizerConfig,
        resolution: Resolution,
        beta: Beta,
        repeat: int,
    ):
        """
        Run one optimizer on all datasets of one condition.

        The results are written to
        ``dirs.optimize / 'Optimizer<case>_repeat<NNN>.parquet'``;
        cases whose output file already exists are skipped.

        Parameters
        ----------
        optimizer
            Optimizer specification.
        resolution
            Temporal resolution.
        beta
            Slope percentile.
        repeat
            Index of the Monte Carlo repetition.
        """
        opt = str(optimizer)
        case = f'{opt}_{resolution}_{beta}'

        path = self.dirs.optimize / f'Optimizer{case}_repeat{repeat:03d}.parquet'
        if path.exists():
            return

        logger.info(
            'optimizing',
            optimizer=opt,
            resolution=resolution,
            beta=beta,
            repeat=repeat,
        )

        generator = cd.Generator(repeat=repeat, resolution=resolution, beta=beta)
        dicts = []

        for d in tqdm(
            generator.iter(),
            total=generator.count,
            desc=f'r{repeat} {case}',
            leave=False,
        ):
            res = optimizer(d)()
            dicts.append({**res, 'repeat': repeat})

        df = pl.from_dicts(dicts, strict=False)
        df.write_parquet(path)

    def _iter(self):
        """
        Run ``_optimize`` for every case, yielding after each one.

        Yields
        ------
        None
        """
        it = itertools.product(
            range(self.repeat), self.optimizers, self.resolution, self.beta
        )

        if self.reverse:
            it = reversed(list(it))

        for repeat, optimizer, resolution, beta in it:
            self._optimize(
                optimizer=optimizer,
                resolution=resolution,
                beta=beta,
                repeat=repeat,
            )
            yield

    def __call__(self):  # ruff: ignore[undocumented-public-method]
        self.dirs.optimize.mkdir(exist_ok=True)

        total = (
            len(self.optimizers) * len(self.resolution) * len(self.beta) * self.repeat
        )
        for _ in tqdm(
            self._iter(),
            total=total,
            desc='Total Cases',
        ):
            pass


if __name__ == '__main__':
    warnings.simplefilter('ignore', TqdmExperimentalWarning)
    warnings.simplefilter('ignore', SingularMatrixWarning)
    app()
