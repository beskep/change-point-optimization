"""Aggregate the optimization results and compute the parameter errors."""

import dataclasses as dc
from typing import TYPE_CHECKING, ClassVar

import matplotlib.pyplot as plt
import polars as pl
import polars.selectors as cs
import seaborn as sns
import structlog

import cpm.optimizer as co
from cpm import evaluate
from cpm.dirs import Dirs  # ruff: ignore[typing-only-first-party-import]
from cpm.evaluate import app

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = structlog.stdlib.get_logger()


@app.command
@dc.dataclass
class Preprocess:
    """Aggregate the optimization results and compute the parameter errors."""

    dirs: Dirs

    def models(self):
        """
        Concatenate the optimization results and assign display names.

        Optimizer names (e.g. ``TRRLinear``) are mapped to the display names
        of ``cpm.optimizer.OPTIMIZERS``.

        Returns
        -------
        polars.DataFrame
            One row per optimizer and dataset, with a row index.

        Raises
        ------
        FileNotFoundError
            If no result file is found in ``dirs.optimize``.
        """
        src = self.dirs.optimize

        if not (files := tuple(src.glob('Optimizer*repeat*.parquet'))):
            raise FileNotFoundError(src)

        models = {
            'IMT0.25': 'IMT',
            'IMT+HuberT+HuberT': 'IMT(Huber)',
            'GridSearch0.25': 'GS(0.25)',
            'GridSearch0.15': 'GS',
            'SegmentedRegression': 'SR',
            'MultiStartTRRLinear': 'TRR(MS)',
            'TRRLinear': 'TRR(Linear)',
            'TRRSoftL1': 'TRR(SoftL1)',
            'InformedStartTRR(Linear)': 'TRR(IS)',
        }

        return (
            pl
            .concat((pl.scan_parquet(x) for x in files), how='diagonal')
            .with_columns(
                pl
                .concat_str(
                    pl.col('optimizer').str.replace_many(models),
                    pl.col('params.linear_model').replace({'OLS': None}),
                    pl.col('params.object').replace({'SSE': None}),
                    separator='+',
                    ignore_nulls=True,
                )
                .str.replace_many(models)
                .alias('optimizer'),
            )
            .sort(
                pl.col('optimizer').replace_strict(co.OPTIMIZERS_ORDER, default=None),
                'optimizer',
                'dataset.resolution',
                'dataset.beta',
                'dataset.repeat',
            )
            .with_row_index()
            .collect()
        )

    @staticmethod
    def error(models: pl.DataFrame):
        """
        Compute the estimation errors of the converged models.

        Parameters
        ----------
        models
            Output of ``models``.

        Returns
        -------
        polars.DataFrame
            Long format with one row per model and parameter (``variable``),
            holding ``ytrue``, ``ypred`` and ``error = ypred - ytrue``.
        """
        params = models.select(cs.starts_with('params.')).columns
        group = [
            'index',
            'repeat',
            'optimizer',
            'dataset.resolution',
            'dataset.beta',
            *params,
        ]
        return (
            models
            .filter(pl.col('converged'))
            .select(*group, cs.ends_with('.true', '.pred'))
            .unpivot(index=group)
            .with_columns(
                pl.col('variable').str.extract_groups(r'^(?<variable>\w+)\.(?<y>\w+)$')
            )
            .unnest('variable')
            .with_columns(
                pl.col('y').replace_strict({'true': 'ytrue', 'pred': 'ypred'})
            )
            .pivot('y', index=[*group, 'variable'], values='value')
            .with_columns((pl.col('ypred') - pl.col('ytrue')).alias('error'))
        )

    def __call__(self):
        """
        Write ``01.models.parquet``, ``02.error.parquet`` and the failed cases.

        Returns
        -------
        models : polars.DataFrame
        error : polars.DataFrame
        """
        output = self.dirs.evaluate
        output.mkdir(exist_ok=True)

        models = self.models()
        error = self.error(models)

        models.write_parquet(output / '01.models.parquet')
        error.write_parquet(output / '02.error.parquet')

        (
            (models)
            .filter(pl.col('converged').not_())
            .write_excel(output / '01.models.failed.xlsx')
        )

        return models, error


@app.command
@dc.dataclass
class Inspect(evaluate.Command):
    """Plot bar charts of the metrics and errors by optimizer and condition."""

    OUTPUT: ClassVar[str] = '03.inspect'
    KWARGS: ClassVar[dict[str, object]] = {
        'y': 'optimizer',
        'col': 'condition',
        'orient': 'h',
        'height': 3.5,
        'margin_titles': True,
        'sharex': 'row',
    }

    def _scan(self, name):
        """``evaluate.Command._scan`` with a ``condition`` (resolution, beta) column."""
        return (
            super()
            ._scan(name)
            .with_columns(
                condition=pl.format('{} beta {}', 'dataset.resolution', 'dataset.beta')
            )
        )

    def _plot(self, data: pl.DataFrame, variables: Sequence[str], name: str, **kwargs):
        """
        Plot bar charts of selected variables.

        Parameters
        ----------
        data
            Long-format data with ``variable`` and ``value`` columns.
        variables
            Variables to plot, one row of panels each.
        name
            Output file name (without extension).
        **kwargs
            Passed to ``seaborn.catplot``.
        """
        logger.info('variables=%s', variables)

        d = data.filter(pl.col('variable').is_in(variables))

        (
            sns
            .catplot(d, x='value', kind='bar', row='variable', **self.KWARGS, **kwargs)
            .set_axis_labels('', '')
            .set_titles(row_template='{row_name}', col_template='{col_name}')
            .savefig(self.output / f'{name}.png')
        )
        plt.close('all')

    def __call__(self):  # ruff: ignore[undocumented-public-method]
        models = (
            self
            ._scan('models')
            .unpivot(
                [
                    'metric.r2',
                    'metric.ols.r2adj',
                    'metric.cvrmse',
                    'metric.nmbe',
                    'duration',
                    'count.lm-fit',
                    'count.obj-eval',
                ],
                index=['optimizer', 'condition'],
            )
            .collect()
        )
        self._plot(models, variables=['metric.r2', 'metric.ols.r2adj'], name='r2')
        self._plot(
            models, variables=['metric.cvrmse', 'metric.nmbe'], name='energy.error'
        )
        self._plot(
            models,
            variables=['duration', 'count.lm-fit', 'count.obj-eval'],
            name='duration',
        )

        error = self._scan('error').rename({'error': 'value'}).collect()
        self._plot(error, variables=['t_h', 't_c'], name='error.cp')
        self._plot(error, variables=['beta_h', 'beta_c'], name='error.beta')

        self._plot(
            error.filter(
                pl.col('optimizer').is_in([
                    'IMT',
                    'IMT(Huber)',
                    'TRR(Linear)',
                    'TRR(SoftL1)',
                    'TRR(MS)',
                ])
            ),
            variables=['t_h', 't_c'],
            name='error.cp.major',
        )


if __name__ == '__main__':
    app()
