"""Evaluate the goodness of fit and convergence of the optimizers."""

import dataclasses as dc

import polars as pl
import polars.selectors as cs
import structlog

from cpm import evaluate
from cpm.evaluate import app

logger = structlog.stdlib.get_logger()


@app.default
@dc.dataclass
class Evaluate(evaluate.Command):
    """Summarise the goodness of fit and convergence of each optimizer."""

    def data(self):
        """
        Load the model metrics and compare them with the reference model.

        Returns
        -------
        polars.DataFrame
        """
        data = (
            self
            ._scan('models')
            .rename(lambda x: x.removeprefix('metric.'))
            .rename({
                'count.lm-fit': 'n.fit',
                'count.obj-eval': 'n.obj',
            })
            .with_columns(
                p=pl.col('optimizer').replace_strict({'Reference': 3}, default=5)
            )
            .collect()
        )
        index = [
            'dataset.beta',
            'dataset.resolution',
            'dataset.station',
            'dataset.year',
            'repeat',
        ]
        reference = (
            (data)
            .filter(pl.col('optimizer') == 'Reference')
            .select(
                *index,
                pl.col('sse').alias('sse.ref'),
                pl.col('rmse').alias('rmse.ref'),
            )
        )
        return (
            (data)
            .join(reference, on=index, how='left', validate='m:1')
            .with_columns(
                (pl.col('sse') / pl.col('sse.ref')).alias('sse.rel'),
                (pl.col('rmse') / pl.col('rmse.ref')).alias('rmse.rel'),
                pl
                .when(pl.col('optimizer') == 'Reference')
                .then(pl.lit(None))
                .otherwise(pl.col('rmse') < pl.col('rmse.ref'))
                .alias('rmse<ref'),
            )
        )

    @staticmethod
    def summary(data: pl.DataFrame):
        """
        Average the metrics of the converged models by optimizer and condition.

        Parameters
        ----------
        data
            Output of ``data``.

        Returns
        -------
        polars.DataFrame
            Mean and standard deviation of the numeric columns, and the
            convergence rate (``None`` for the reference model).
        """
        index = ['optimizer', 'dataset.resolution', 'dataset.beta']
        convergence_rate = (
            data
            .with_columns(
                pl
                .when(pl.col('optimizer') == 'Reference')
                .then(pl.lit(None))
                .otherwise(pl.col('converged').cast(pl.Float64))
                .alias('converged')
            )
            .group_by(index)
            .agg(pl.mean('converged'))
        )
        return (
            data
            .filter(pl.col('converged'))
            .group_by(index)
            .agg(
                pl.col('index').first(),
                cs.numeric().mean().name.suffix('.mean'),
                cs.numeric().std().name.suffix('.std'),
            )
            .with_columns(
                pl
                .when(pl.col('optimizer') == 'Reference')
                .then(pl.lit(None))
                .otherwise(pl.col('sse.rel.mean'))
                .alias('sse.rel.mean'),
                pl
                .when(pl.col('optimizer') == 'Reference')
                .then(pl.lit(None))
                .otherwise(pl.col('rmse.rel.mean'))
                .alias('rmse.rel.mean'),
            )
            .fill_nan(None)
            .sort('index')
            .join(convergence_rate, on=index, how='left')
        )

    @staticmethod
    def table(summary: pl.DataFrame):
        """
        Format the summary as a table of ``mean+-std`` strings.

        Parameters
        ----------
        summary
            Output of ``summary``.

        Returns
        -------
        polars.DataFrame
        """
        return summary.select(
            'dataset.resolution',
            'dataset.beta',
            'optimizer',
            pl.format(
                '{}+-{}',
                pl.col('r2.mean').map_elements(lambda x: f'{x:.3f}'),
                pl.col('r2.std').map_elements(lambda x: f'{x:.3f}'),
            ).alias('r2'),
            pl.format(
                '{}%+-{}%',
                pl.col('cvrmse.mean').map_elements(lambda x: f'{100 * x:.2f}'),
                pl.col('cvrmse.std').map_elements(lambda x: f'{100 * x:.2f}'),
            ).alias('cvrmse'),
            pl.format(
                '{}+-{}',
                pl.col('sse.rel.mean').map_elements(lambda x: f'{x:.4f}'),
                pl.col('sse.rel.std').map_elements(lambda x: f'{x:.4f}'),
            ).alias('sse.rel'),
            pl.format(
                '{}+-{}',
                pl.col('n.fit.mean').map_elements(lambda x: f'{x:.1f}'),
                pl.col('n.fit.std').map_elements(lambda x: f'{x:.1f}'),
            ).alias('n.fit'),
            pl
            .col('converged')
            .map_elements(lambda x: f'{100 * x:.2f}%')
            .alias('convergence.rate'),
        )

    def __call__(self):
        """
        Write the data, summary and tables (``03.energy.*``).

        Returns
        -------
        polars.DataFrame
            Formatted table.
        """
        data = self.data()
        summary = self.summary(data)
        table = self.table(summary)

        data.write_parquet(self.dirs.evaluate / '03.energy.parquet')
        summary.write_csv(self.dirs.evaluate / '03.energy.summary.csv')
        table.write_csv(self.dirs.evaluate / '03.energy.table.csv')
        txt = evaluate.typst_table(table)
        self.dirs.evaluate.joinpath('03.energy.table.txt').write_text(txt)

        return table


if __name__ == '__main__':
    app()
