"""Supplementary analyses and figures."""

import dataclasses as dc
import functools
import itertools
from typing import TYPE_CHECKING, ClassVar

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import polars.selectors as cs
import rich
import seaborn as sns
import skmisc.loess
import structlog
from matplotlib.figure import Figure
from matplotlib.layout_engine import ConstrainedLayoutEngine
from matplotlib.ticker import MaxNLocator
from tqdm.rich import tqdm

import cpm.dataset as cd
from cpm import evaluate
from cpm.dataset import Beta  # ruff:ignore[typing-only-first-party-import]
from cpm.dirs import Dirs  # ruff:ignore[typing-only-first-party-import]
from cpm.evaluate import Color, app

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes

EXP_REL_SSE = (365 - 5) / (365 - 3)
"""Expected Relative SSE of a daily 5P CPM, (n-5)/(n-3)."""

logger = structlog.stdlib.get_logger()


@app.command
class SrBracket(evaluate.Command):
    """Report how often the SR change-points lie within their bracketing samples."""

    def __call__(self):  # ruff: ignore[undocumented-public-method]
        return (
            self
            ._scan('models')
            .filter(pl.col('optimizer') == 'SR')
            .unpivot(
                ['sr.t_h.in_bracket', 'sr.t_c.in_bracket'],
                index=['dataset.resolution', 'dataset.beta'],
            )
            .group_by('dataset.resolution', 'dataset.beta')
            .agg(pl.mean('value') * 100)
            .sort(pl.all())
            .collect()
        )


@app.command
class MuggeoFailure(evaluate.Command):
    """Count the failures of Muggeo's method by cause."""

    def __call__(self):  # ruff: ignore[undocumented-public-method]
        return (
            self
            ._scan('models')
            .filter(pl.col('optimizer') == 'Muggeo', pl.col('converged').not_())
            .with_columns(
                pl.col('error').replace_strict({
                    'Bound.Th': 'bound',
                    'Bound.Tc': 'bound',
                    'Numerical': 'lib',
                })
            )
            .group_by('dataset.resolution', 'error')
            .agg(pl.len())
            .collect()
            .pivot('error', index='dataset.resolution', values='len', sort_columns=True)
            .sort(pl.all())
        )


@app.command
class MuggeoBaseSlope(evaluate.Command):
    """Summarise the baseline-segment slope estimated by Muggeo's method."""

    def __call__(self):  # ruff: ignore[undocumented-public-method]
        v = pl.col('muggeo.beta_baseline')
        return (
            self
            ._scan('models')
            .filter(pl.col('optimizer') == 'Muggeo')
            .with_columns(v * 1000)  # kWh/m²℃ -> Wh/m²℃
            .group_by('dataset.resolution', 'dataset.beta')
            .agg(
                v.mean().alias('avg'),
                v.std().alias('std'),
            )
            .with_columns(cs.float().round(2))
            .collect()
            .sort(pl.all())
        )


@app.command
class BiasPlot(evaluate.Command):
    """Plot the mean change-point error with confidence intervals."""

    OPTIMIZERS: ClassVar[tuple[str, ...]] = ('IMT', 'TRR(Linear)', 'TRR(MS)', 'TRR(IS)')
    XLIM: ClassVar[dict[str, tuple[float, float]]] = {
        'd': (-0.35, 0.2),
        'mo': (-0.5, 0.7),
    }

    def data(self):
        """
        Load the mean errors and confidence half-widths of ``OPTIMIZERS``.

        Returns
        -------
        polars.DataFrame
        """
        order = {v: i for i, v in enumerate(self.OPTIMIZERS)}
        return (
            pl
            .scan_parquet(self.dirs.evaluate / '04.03.CP.summary.parquet')
            .filter(pl.col('optimizer').is_in(self.OPTIMIZERS))
            .select(
                'dataset.resolution',
                'dataset.beta',
                'optimizer',
                't_h.mbe',
                't_h.ci.half',
                't_c.mbe',
                't_c.ci.half',
            )
            .rename(lambda x: x.removeprefix('dataset.').removesuffix('.half'))
            .unpivot(index=['resolution', 'beta', 'optimizer'])
            .with_columns(
                pl.col('variable').str.extract_groups(r'^(?<cp>t_[hc])\.(?<stat>\w+)$')
            )
            .unnest('variable')
            .with_columns(
                pl.format(
                    '${}({})$.{}',
                    pl.col('cp').replace_strict({'t_h': 'T_h', 't_c': 'T_c'}),
                    pl.col('beta').replace_strict({
                        '50%': r'\beta_{0.5}',
                        '10%': r'\beta_{0.1}',
                    }),
                    'stat',
                ).alias('variable')
            )
            .sort(pl.col('optimizer').replace_strict(order))
            .collect()
        )

    def _plot_bar(
        self,
        ax: Axes,
        data: pl.DataFrame,
        *,
        groups: Sequence[str],
        height: float = 0.8,
    ):
        """
        Draw horizontal bars of the mean error with error bars.

        Parameters
        ----------
        ax
            Target axes.
        data
            Output of ``data`` for one resolution.
        groups
            Bar groups (change-point and beta), drawn side by side.
        height
            Total height of the bars of one optimizer.
        """
        ybase = np.arange(len(self.OPTIMIZERS))
        height /= len(groups)
        locator = MaxNLocator(nbins=12, steps=[1, 2, 2.5, 5, 10])

        for idx, g in enumerate(groups):
            offset = (idx + 0.5 - len(groups) * 0.5) * height
            width = data.filter(pl.col('variable') == f'{g}.mbe')['value']
            xerr = data.filter(pl.col('variable') == f'{g}.ci')['value']

            color = Color.H if 'T_h' in g else Color.C
            alpha = 0.4 if 'beta_{0.1}' in g else 1

            ax.barh(
                y=ybase + offset,
                width=width,
                height=height,
                xerr=xerr,
                color=(color, alpha),
                label=g,
                error_kw={
                    'elinewidth': 0.6,
                    'capsize': 2,
                    'capthick': 0.6,
                    'ecolor': '.1',
                },
                edgecolor=color,
                linewidth=0.8,
            )

            ax.xaxis.set_major_locator(locator)
            ax.set_yticks(ybase, labels=self.OPTIMIZERS)
            ax.set_xlabel('Change-Point Bias [°C]', labelpad=-1)

    def plot(self, *, clip: bool = True):
        """
        Plot daily and monthly results (``07.bias*``).

        Parameters
        ----------
        clip
            Limit the x-axis to ``XLIM``. The bars of TRR(Linear) for T_h on
            monthly datasets, which exceed the limit, are annotated with
            their values.
        """
        raw = self.data()
        groups = (
            r'$T_h(\beta_{0.5})$',
            r'$T_h(\beta_{0.1})$',
            r'$T_c(\beta_{0.5})$',
            r'$T_c(\beta_{0.1})$',
        )

        fig = Figure(figsize=(16, 12, 'cm'))
        axes = fig.subplots(2, 1, sharey=True, squeeze=False)

        arrowprops = {'arrowstyle': '->', 'lw': 0.6, 'shrinkA': 0, 'shrinkB': 2}
        annotation_style = {
            'textcoords': 'offset points',
            'size': 'small',
            'va': 'center',
            'path_effects': [pe.Stroke(linewidth=2, foreground='w'), pe.Normal()],
        }
        ax: Axes
        for ax, resolution in zip(axes.ravel(), ('d', 'mo'), strict=True):
            data = raw.filter(pl.col('resolution') == resolution)
            self._plot_bar(ax, data, groups=groups)

            ax.set_title('Daily Datasets' if resolution == 'd' else 'Monthly Datasets')

            if clip:
                ax.set_xlim(self.XLIM[resolution])

            if resolution == 'd':
                ax.set_yinverted(inverted=True)
            else:
                ax.legend(ncols=2, fontsize='small', borderaxespad=1)

                if not clip:
                    continue

                annot = (
                    raw
                    .filter(
                        pl.col('optimizer') == 'TRR(Linear)',
                        pl.col('resolution') == 'mo',
                        pl.col('cp') == 't_h',
                    )
                    .with_columns(pl.col('value').round(3))
                    .pivot('stat', index='beta', values='value')
                    .with_columns(pl.format(r'${}\pm{}$', 'mbe', 'ci').alias('text'))
                    .rows_by_key('beta', named=True, unique=True)
                )

                xmin = self.XLIM['mo'][0]
                ax.annotate(
                    # beta_0.5
                    text=annot['50%']['text'],
                    xy=(xmin, 0.72),
                    xytext=(20, 8),
                    arrowprops={**arrowprops, 'connectionstyle': 'arc3,rad=0.15'},
                    **annotation_style,
                )
                ax.annotate(
                    # beta_0.1
                    text=annot['10%']['text'],
                    xy=(xmin, 0.88),
                    xytext=(20, -6),
                    arrowprops={**arrowprops, 'connectionstyle': 'arc3,rad=-0.15'},
                    **annotation_style,
                )

        suffix = '.clip' if clip else ''
        fig.savefig(self.dirs.evaluate / f'07.bias{suffix}.svg')
        fig.savefig(self.dirs.evaluate / f'07.bias{suffix}.pdf')

    def __call__(self):  # ruff: ignore[undocumented-public-method]
        with plt.rc_context({
            'axes.grid': False,
            'axes.xmargin': 0.01,
            'figure.constrained_layout.h_pad': 2 / 72,
            'figure.constrained_layout.w_pad': 2 / 72,
            'ytick.left': False,
            'ytick.right': False,
        }):
            self.plot(clip=True)
            self.plot(clip=False)


@app.command
@dc.dataclass
class SseCp(evaluate.Command):
    """
    Plot LOESS curves of the change-point error against relative SSE.

    Parameters
    ----------
    span
        LOESS span.
    qcut
        Quantile of relative SSE trimmed from each tail before smoothing.
    """

    span: float = 2 / 3
    qcut: float = 0.01

    def data(self):
        """
        Change-point errors and relative SSE of IMT and TRR(MS) (daily data).

        Returns
        -------
        polars.DataFrame
        """
        return (
            pl
            .scan_parquet(self.dirs.evaluate / '03.energy.parquet')
            .filter(
                pl.col('dataset.resolution') == 'd',
                pl.col('optimizer').is_in(['IMT', 'TRR(MS)']),
            )
            .with_columns(
                pl
                .col('sse.rel')
                .qcut([self.qcut, 1 - self.qcut], labels=['low', 'mid', 'high'])
                .over(['optimizer', 'dataset.beta'])
                .alias('qcut')
            )
            .filter(pl.col('qcut') == 'mid')
            .with_columns(
                (pl.col('t_h.pred') - pl.col('t_h.true')).alias('$T_h$'),
                (pl.col('t_c.pred') - pl.col('t_c.true')).alias('$T_c$'),
            )
            .unpivot(['$T_h$', '$T_c$'], index=['optimizer', 'dataset.beta', 'sse.rel'])
            .with_columns(
                pl
                .col('dataset.beta')
                .replace_strict({'10%': r'$\beta_{0.1}$', '50%': r'$\beta_{0.5}$'})
                .alias('beta')
            )
            .collect()
        )

    def fit(self, data: pl.DataFrame):
        """
        Fit a LOESS curve of the error against relative SSE.

        Parameters
        ----------
        data
            Data of one group with ``sse.rel`` and ``value`` columns.

        Returns
        -------
        polars.DataFrame
            Fitted values with standard errors and 95% confidence limits.
        """
        data = data.sort('sse.rel')
        x = data['sse.rel'].to_numpy()
        model = skmisc.loess.loess(x, data['value'].to_numpy(), span=self.span)
        model.fit()

        pred = model.predict(x, stderror=True)
        ci = pred.confidence(0.05)

        return pl.DataFrame({
            'x': x,
            'y': pred.values,
            'se': pred.stderr,
            'ymin': ci.lower,
            'ymax': ci.upper,
        })

    @functools.cached_property
    def loess(self):
        """
        LOESS fits by optimizer, beta and change-point.

        Cached in ``05.LOESS.parquet``.
        """
        if (path := self.dirs.evaluate / '05.LOESS.parquet').exists():
            return pl.read_parquet(path)

        data = self.data()

        def it():
            for (opt, b, v), df in data.group_by('optimizer', 'beta', 'variable'):
                yield self.fit(df).with_columns(
                    pl.lit(opt).alias('optimizer'),
                    pl.lit(b).alias('beta'),
                    pl.lit(v).alias('variable'),
                )

        loess = pl.concat(it())
        loess.write_parquet(path)
        return loess

    @staticmethod
    def _plot(data, *_, **__):
        """Draw LOESS curves and confidence bands; for ``FacetGrid.map_dataframe``."""
        data = pl.from_pandas(data)

        ax = plt.gca()
        for _, df in data.group_by('variable'):
            ax.fill_between(
                df['x'].to_numpy(),
                df['ymin'].to_numpy(),
                df['ymax'].to_numpy(),
                color='slategray',
                alpha=0.15,
                linewidth=0,
                label='_nolegend_',
            )

        sns.lineplot(
            data,
            x='x',
            y='y',
            hue='variable',
            hue_order=['$T_h$', '$T_c$'],
            palette=evaluate.Color,
            ax=ax,
            lw=1.5,
            alpha=0.8,
        )

    def plot(self):
        """
        Plot the LOESS curves in a facet grid.

        Returns
        -------
        seaborn.FacetGrid
        """
        grid = (
            sns
            .FacetGrid(
                self.loess,
                row='beta',
                col='optimizer',
                despine=False,
                sharey='row',
                aspect=16 / 9,
            )
            .map_dataframe(self._plot)
            .set(xmargin=0.02, ylim=(-0.6, 0.6))
            .set_titles('{col_name} on {row_name} condition')
            .set_axis_labels(x_var='Relative SSE', labelpad=-1)
            .set_axis_labels(
                y_var='Change-Point Bias [°C]', stretch='condensed', labelpad=2.5
            )
        )

        axes = list(grid.axes_dict.values())
        axes[0].legend(fontsize='small', labelspacing=0.2)

        locator = MaxNLocator(nbins='auto', steps=[1, 2])
        for ax in axes:
            ax.axhline(0, lw=0.8, c='k', alpha=0.4, zorder=0)
            ax.axvline(EXP_REL_SSE, lw=0.8, c='k', alpha=0.5, ls='--', zorder=0)
            ax.yaxis.set_major_locator(locator)

        axes[0].annotate(
            f'$\\dfrac{{n-5}}{{n-3}}\\approx{EXP_REL_SSE:.4f}$',
            (EXP_REL_SSE, -0.6),
            xytext=(2.5, 15),
            textcoords='offset points',
            fontsize='small',
            c='.2',
        )

        grid.figure.set_size_inches(18 / 2.54, 9 / 2.54)
        ConstrainedLayoutEngine().execute(grid.figure)

        return grid

    def __call__(self):
        """Write the LOESS extrema and the figure (``05.*``)."""
        (
            self.loess
            .filter(pl.col('x') <= (365 - 5) / (365 - 3))
            .group_by('optimizer', 'beta', 'variable')
            .agg(pl.max('y').alias('ymax'), pl.min('y').alias('ymin'))
            .sort(pl.all())
            .write_csv(self.dirs.evaluate / '05.LOESS.csv')
        )

        with plt.rc_context({
            'axes.grid': False,
            'xtick.top': False,
            'ytick.right': False,
        }):
            grid = self.plot()
            grid.savefig(self.dirs.evaluate / '05.relative-sse-bias.svg')
            grid.savefig(self.dirs.evaluate / '05.relative-sse-bias.pdf')


@dc.dataclass(frozen=True)
class _DatasetGenerator(cd.Generator):
    """
    Generator computing annual heating and cooling energy by two methods.

    ``daily`` computes the energy from daily temperature; ``monthly``
    computes it from monthly mean temperature. Both are expressed as the sum
    of monthly mean daily energy.
    """

    resolution: cd.Resolution = 'd'

    def calculate(self, d: cd.Dataset):
        """
        Compute the heating and cooling energy of a dataset by both methods.

        Parameters
        ----------
        d
            Dataset providing the reference parameters.

        Returns
        -------
        polars.DataFrame
            Two rows (``daily``, ``monthly``) with ``E_h`` and ``E_c``.
        """
        eh = (
            d.beta_h * pl.max_horizontal(0, pl.lit(d.t_h) - pl.col('temperature'))
        ).alias('E_h')
        ec = (
            d.beta_c * pl.max_horizontal(0, pl.col('temperature') - pl.lit(d.t_c))
        ).alias('E_c')

        temperature = self._temperature(station=d.station, year=d.year, resolution='d')

        monthly = (
            temperature
            .group_by_dynamic('date', every='1mo')
            .agg(pl.mean('temperature'))
            .with_columns(eh, ec)
            .select(pl.sum('E_h', 'E_c'), pl.lit('monthly').alias('method'))
        )
        daily = (
            temperature
            .with_columns(eh, ec)
            .group_by_dynamic('date', every='1mo')
            .agg(pl.mean('E_h', 'E_c'))
            .select(pl.sum('E_h', 'E_c'), pl.lit('daily').alias('method'))
        )
        return pl.concat([daily, monthly]).with_columns(
            station=d.station, year=d.year, repeat=d.repeat
        )

    def iter(self):
        """
        Yield the result of ``calculate`` for every station-year.

        Yields
        ------
        polars.DataFrame
        """
        for dataset in super().iter():
            yield self.calculate(dataset)


@app.command
@dc.dataclass(frozen=True)
class Jensen:
    """
    Quantify the energy difference from monthly averaging (Jensen's inequality).

    Parameters
    ----------
    repeat
        Number of Monte Carlo repetitions.
    """

    _: dc.KW_ONLY
    dirs: Dirs
    repeat: int = 20

    @staticmethod
    def _calculate(beta: Beta, repeat: int):
        """
        Compute both energies for all datasets of one beta and repetition.

        Parameters
        ----------
        beta
            Slope percentile.
        repeat
            Index of the Monte Carlo repetition.

        Returns
        -------
        polars.DataFrame
        """
        generator = _DatasetGenerator(repeat=repeat, beta=beta)
        return pl.concat(generator.iter()).with_columns(
            pl.lit(beta).alias('beta'), pl.lit(repeat).alias('repeat')
        )

    def calculate(self):
        """
        Compute both energies for all conditions (cached in ``06.jensen.parquet``).

        Returns
        -------
        polars.DataFrame
        """
        if (path := self.dirs.evaluate / '06.jensen.parquet').exists():
            return pl.read_parquet(path)

        it = itertools.starmap(
            self._calculate, itertools.product(('10%', '50%'), range(self.repeat))
        )
        data = pl.concat(tqdm(it, total=self.repeat * 2))
        data.write_parquet(path)
        return data

    def __call__(self):
        """
        Write the counts of zero energy and the statistics of the difference.

        Returns
        -------
        polars.DataFrame
            Daily and monthly energy of each case in wide format.
        """
        case = ['station', 'year', 'repeat']
        data = (
            self
            .calculate()
            .unpivot(['E_h', 'E_c'], index=['beta', 'method', *case])
            .pivot('method', index=['beta', *case, 'variable'], values='value')
        )
        zero = (
            data
            .group_by('beta', 'variable')
            .agg(
                pl.len(),
                (pl.col('daily') == 0).sum(),
                (pl.col('monthly') == 0).sum(),
            )
            .sort(pl.all())
        )
        stat = (
            data
            .with_columns((pl.col('daily') - pl.col('monthly')).alias('delta'))
            .unpivot(
                ['daily', 'monthly', 'delta'],
                variable_name='stat',
                index=['beta', 'variable'],
            )
            .group_by('beta', 'variable', 'stat', maintain_order=True)
            .agg(
                pl.col('value').mean().alias('mean'),
                pl.col('value').std().alias('std'),
                pl.col('value').min().alias('min'),
                pl.col('value').median().alias('median'),
                pl.col('value').max().alias('max'),
            )
        )

        zero.write_csv(self.dirs.evaluate / '06.jensen.zero.csv')
        stat.write_csv(self.dirs.evaluate / '06.jensen.stats.csv')

        return data


@app.default
def run(dirs: Dirs):
    """Run commands in order."""
    console = rich.get_console()

    for cls in (SrBracket, MuggeoFailure, MuggeoBaseSlope, BiasPlot, SseCp, Jensen):
        logger.info(cls.__name__)
        if (r := cls(dirs=dirs)()) is not None:
            console.print(r)


if __name__ == '__main__':
    app()
