"""Evaluate the change-point errors and the resulting heating/cooling energy errors."""

import dataclasses as dc
import functools
from typing import Literal

import polars as pl
import polars.selectors as cs
import rich
import structlog
from scipy.stats import ttest_1samp
from tqdm.rich import tqdm

from cpm import evaluate
from cpm.dirs import Dirs  # ruff: ignore[typing-only-first-party-import]
from cpm.evaluate import app
from cpm.optimizer import OPTIMIZERS_ORDER

logger = structlog.stdlib.get_logger()


def t_test(
    data: pl.DataFrame,
    *,
    value: str = 'error',
    popmean: float = 0,
    alternative: Literal['two-sided', 'less', 'greater'] = 'two-sided',
    confidence_level: float = 0.95,
):
    """
    One-sample t-test of a group, for ``polars.GroupBy.map_groups``.

    Parameters
    ----------
    data
        Data of one group.
    value
        Column to test.
    popmean
        Expected mean under the null hypothesis.
    alternative
        Alternative hypothesis.
    confidence_level
        Confidence level of the confidence interval.

    Returns
    -------
    polars.DataFrame
        First row of ``data`` without ``value``, with the p-value, degrees of
        freedom and the confidence interval of the mean
        (``ci.low``, ``ci.high``, ``ci.half``).
    """
    array = data.select(value).to_numpy().ravel()
    r = ttest_1samp(array, popmean=popmean, alternative=alternative)
    ci = r.confidence_interval(confidence_level)
    return (
        data
        .head(n=1)
        .drop(value)
        .with_columns(
            pl.lit(r.pvalue).alias('p-value'),
            pl.lit(r.df).alias('df'),
            pl.lit(ci.low).alias('ci.low'),
            pl.lit(ci.high).alias('ci.high'),
            pl.lit((ci.high - ci.low) / 2.0).alias('ci.half'),
        )
    )


def _ci_columns(v: str):
    return [f'{v}.ci.low', f'{v}.ci.high', f'{v}.ci.half', f'{v}.p-value']


def _fmt2(value: float):
    return f'{value:.2f}'


def _fmt3(value: float):
    return f'{value:.3f}'


@app.command
@dc.dataclass
class ChangePointTTest(evaluate.Command):
    """Test whether the mean change-point error of each optimizer is zero."""

    def __call__(self):
        """
        Write the t-test results (``04.01.CP.t-test.*``).

        Returns
        -------
        polars.DataFrame
        """
        errors = (
            self
            ._scan('error')
            .filter(
                pl.col('optimizer') != 'Reference',
                pl.col('variable').is_in(['t_h', 't_c']),
            )
            .collect()
        )
        index = ['dataset.resolution', 'dataset.beta', 'optimizer', 'variable']
        t = (
            errors
            .select(*index, 'error')
            .group_by(index, maintain_order=True)
            .map_groups(t_test)
        )
        t.write_parquet(self.dirs.evaluate / '04.01.CP.t-test.parquet')
        t.write_csv(self.dirs.evaluate / '04.01.CP.t-test.csv')
        return t


@app.command
class PredictEnergy(evaluate.Command):
    """Calculate heating and cooling energy from the parameters of each optimizer."""

    @staticmethod
    def _energy(mode: Literal['h', 'c']):
        """
        Expression of the heating (``E_h``) or cooling (``E_c``) energy.

        Parameters
        ----------
        mode
            Heating or cooling.

        Returns
        -------
        polars.Expr
        """
        beta = pl.col(f'beta_{mode}.pred')
        delta = pl.col(f't_{mode}.pred') - pl.col('temperature')
        s = 1 if mode == 'h' else -1
        return (beta * pl.max_horizontal(0, s * delta)).alias(f'E_{mode}')

    @staticmethod
    def _check_len(data: pl.DataFrame):
        """
        Check that every case has a full year of temperature data.

        Parameters
        ----------
        data
            Data with ``dataset.resolution`` and ``len`` columns.

        Raises
        ------
        ValueError
            If any case has fewer than 365 daily or 12 monthly values.
        """
        insufficient = (
            (data)
            .with_columns(
                pl
                .col('dataset.resolution')
                .replace_strict({'d': 365, 'mo': 12})
                .alias('required')
            )
            .filter(pl.col('len') < pl.col('required'))
        )

        if insufficient.height:
            raise ValueError(insufficient)

    @functools.cached_property
    def temperature(self):
        """Daily and monthly mean temperature of every station-year."""
        daily = (
            pl
            .scan_parquet('data/daily-outdoor-temperature.parquet')
            .select('station', 'year', 'date', 'temperature')
            .rename({'station': 'dataset.station', 'year': 'dataset.year'})
            .with_columns(pl.lit('d').alias('dataset.resolution'))
            .collect()
        )
        monthly = (
            daily
            .group_by_dynamic(
                'date', every='1mo', group_by=['dataset.station', 'dataset.year']
            )
            .agg(pl.mean('temperature'))
            .with_columns(pl.lit('mo').alias('dataset.resolution'))
        )

        return pl.concat([daily, monthly])

    @functools.cached_property
    def models(self):
        """Estimated parameters of the converged models."""
        return (
            self
            ._scan('models')
            .filter(pl.col('converged'))
            .select(
                'optimizer',
                'repeat',
                cs.starts_with('dataset'),
                cs.ends_with('.pred'),
            )
            .collect()
        )

    def _calculate(self, optimizer: str):
        """
        Annual baseline, heating, cooling and total energy of one optimizer.

        Parameters
        ----------
        optimizer
            Optimizer display name.

        Returns
        -------
        polars.DataFrame
            One row per dataset with ``E_b``, ``E_h``, ``E_c`` and ``E``.
        """
        return (
            self.models
            .filter(pl.col('optimizer') == optimizer)
            .join(
                self.temperature,
                on=['dataset.resolution', 'dataset.station', 'dataset.year'],
            )
            .with_columns(
                self._energy('h'),
                self._energy('c'),
            )
            .rename({'baseline.pred': 'E_b'})
            .with_columns(pl.sum_horizontal('E_b', 'E_h', 'E_c').alias('E'))
            .group_by([
                'optimizer',
                'dataset.resolution',
                'dataset.beta',
                'dataset.station',
                'dataset.year',
                'repeat',
            ])
            .agg(pl.sum('E_b', 'E_h', 'E_c', 'E'))
        )

    def __call__(self):
        """
        Write the predicted energy (``04.02.energy.*``).

        Returns
        -------
        polars.DataFrame
        """
        optimizers = self.models['optimizer'].unique(maintain_order=True).to_list()
        data = pl.concat([self._calculate(x) for x in tqdm(optimizers)]).sort(pl.all())
        data.write_parquet(self.dirs.evaluate / '04.02.energy.parquet')
        data.describe(interpolation='linear').write_csv(
            self.dirs.evaluate / '04.02.energy.summary.csv'
        )

        return data


@app.command
@dc.dataclass
class EnergyStat(evaluate.Command):
    """Compare the predicted energy of each optimizer with the reference model."""

    def __call__(self):
        """
        Write the energy error statistics (``04.02.energy.stat.*``).

        Errors are relative to the reference model of the same dataset;
        datasets where either value is zero are excluded. CV(RMSE), NMBE and
        the confidence interval are normalized by the mean reference energy.

        Returns
        -------
        polars.DataFrame
        """
        index = [
            'dataset.resolution',
            'dataset.beta',
            'dataset.station',
            'dataset.year',
            'repeat',
        ]
        source = (
            pl
            .scan_parquet(self.dirs.evaluate / '04.02.energy.parquet')
            .unpivot(index=['optimizer', *index])
            .collect()
        )
        reference = (
            source
            .filter(pl.col('optimizer') == 'Reference')
            .drop('optimizer')
            .rename({'value': 'reference'})
        )
        data = (
            source
            .join(reference, on=[*index, 'variable'], how='left', validate='m:1')
            .filter(pl.col('value') != 0, pl.col('reference') != 0)
            .with_columns((pl.col('value') - pl.col('reference')).alias('error'))
        )

        index = ['dataset.resolution', 'dataset.beta', 'optimizer', 'variable']
        agg = (
            data
            .group_by(index)
            .agg(
                pl.mean('reference').alias('reference.mean'),
                pl.col('error').pow(2).mean().sqrt().alias('rmse'),
                pl.mean('error').alias('mbe'),
            )
            .with_columns(
                (pl.col('rmse') / pl.col('reference.mean')).alias('cvrmse'),
                (pl.col('mbe') / pl.col('reference.mean')).alias('nmbe'),
            )
        )

        t = data.select(*index, 'error').group_by(index).map_groups(t_test)
        stats = (
            agg
            .join(t, on=index, validate='1:1')
            .with_columns(
                pl.col('ci.low', 'ci.high', 'ci.half') / pl.col('reference.mean')
            )
            .sort(index)
        )

        stats.write_parquet(self.dirs.evaluate / '04.02.energy.stat.parquet')
        stats.write_csv(self.dirs.evaluate / '04.02.energy.stat.csv')

        return stats


@app.command
@dc.dataclass
class Summary(evaluate.Command):
    """Summarise the change-point and energy errors into the paper table."""

    def stats(self):
        """
        RMSE and MBE of the change-point errors.

        Returns
        -------
        polars.DataFrame
        """
        return (
            self
            ._scan('error')
            .filter(
                pl.col('optimizer') != 'Reference',
                pl.col('variable').is_in(['t_h', 't_c']),
            )
            .group_by(['dataset.resolution', 'dataset.beta', 'optimizer', 'variable'])
            .agg(
                pl.col('error').pow(2).mean().sqrt().alias('rmse'),
                pl.mean('error').alias('mbe'),
            )
            .collect()
        )

    def summary(self):
        """
        Join the change-point statistics, t-tests and energy statistics.

        Requires the outputs of ``ChangePointTTest`` and ``EnergyStat``.

        Returns
        -------
        polars.DataFrame
            One row per resolution, beta and optimizer.
        """
        index = ['dataset.resolution', 'dataset.beta', 'optimizer']
        index_var = [*index, 'variable']

        stats = self.stats()
        ttest = pl.read_parquet(self.dirs.evaluate / '04.01.CP.t-test.parquet')
        energy = (
            pl
            .read_parquet(self.dirs.evaluate / '04.02.energy.stat.parquet')
            .unpivot(
                ['cvrmse', 'nmbe', 'ci.low', 'ci.high', 'ci.half', 'p-value'],
                index=index_var,
                variable_name='stat',
            )
            .with_columns(pl.format('{}.{}', 'variable', 'stat').alias('variable'))
            .pivot('variable', index=index, values='value')
        )
        return (
            stats
            .join(ttest, on=index_var, how='left', validate='1:1')
            .select(*index_var, pl.all().exclude(index_var))
            .unpivot(
                ['rmse', 'mbe', 'ci.low', 'ci.high', 'ci.half', 'p-value'],
                index=index_var,
                variable_name='stat',
            )
            .with_columns(pl.format('{}.{}', 'variable', 'stat').alias('on'))
            .pivot('on', index=index, values='value')
            .join(energy, on=index, how='left', validate='1:1')
            .select(
                *index,
                't_h.rmse',
                't_h.mbe',
                *_ci_columns('t_h'),
                't_c.rmse',
                't_c.mbe',
                *_ci_columns('t_c'),
                'E_h.cvrmse',
                'E_h.nmbe',
                *_ci_columns('E_h'),
                'E_c.cvrmse',
                'E_c.nmbe',
                *_ci_columns('E_c'),
                'E_b.cvrmse',
                'E_b.nmbe',
                *_ci_columns('E_b'),
            )
            .sort(
                'dataset.resolution',
                'dataset.beta',
                pl.col('optimizer').replace_strict(OPTIMIZERS_ORDER, default=None),
                descending=[False, True, False],
            )
        )

    def __call__(self):
        """Write the summary (``04.03.CP.summary.*``) and the Typst table."""
        data = self.summary()
        data.write_parquet(self.dirs.evaluate / '04.03.CP.summary.parquet')
        data.write_csv(self.dirs.evaluate / '04.03.CP.summary.csv')

        table = data.select(
            'dataset.resolution',
            'dataset.beta',
            'optimizer',
            # CP
            pl.col('t_h.rmse').map_elements(_fmt3),
            pl.format(
                '{}+-{}',
                pl.col('t_h.mbe').map_elements(_fmt2),
                pl.col('t_h.ci.half').map_elements(_fmt2),
            ).alias('t_h.mbeci'),
            pl.col('t_c.rmse').map_elements(_fmt3),
            pl.format(
                '{}+-{}',
                pl.col('t_c.mbe').map_elements(_fmt2),
                pl.col('t_c.ci.half').map_elements(_fmt2),
            ).alias('t_c.mbeci'),
            # energy
            pl.col('E_h.cvrmse').mul(100).map_elements(_fmt2),
            pl.format(
                '{}+-{}',
                pl.col('E_h.nmbe').mul(100).map_elements(_fmt2),
                pl.col('E_h.ci.half').mul(100).map_elements(_fmt2),
            ).alias('E_h.mbeci'),
            pl.col('E_c.cvrmse').mul(100).map_elements(_fmt2),
            pl.format(
                '{}+-{}',
                pl.col('E_c.nmbe').mul(100).map_elements(_fmt2),
                pl.col('E_c.ci.half').mul(100).map_elements(_fmt2),
            ).alias('E_c.mbeci'),
        )
        txt = evaluate.typst_table(table).replace('-0.00+-', '0.00+-')
        self.dirs.evaluate.joinpath('04.03.CP.table.txt').write_text(txt)


@app.default
def run(dirs: Dirs):
    """Run commands in order."""
    console = rich.get_console()

    for cls in (ChangePointTTest, PredictEnergy, EnergyStat, Summary):
        logger.info(cls.__name__)
        if (r := cls(dirs=dirs)()) is not None:
            console.print(r)


if __name__ == '__main__':
    app()
