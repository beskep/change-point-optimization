"""Common components of the evaluation scripts (``scripts/s03evaluate``)."""

import dataclasses as dc
import enum
import functools
from typing import ClassVar, Literal

import polars as pl
from cyclopts.config import Toml

from cpm.dirs import Dirs  # ruff: ignore[typing-only-first-party-import]
from cpm.optimizer import OPTIMIZERS_ORDER
from cpm.utils import App


class Color(enum.StrEnum):
    """Colours of the heating (``H``) and cooling (``C``) change-points."""

    # Medium contrast color scheme by Paul Tol
    # https://sronpersonalpages.nl/~pault/
    H = '#994455'
    C = '#6699CC'


@dc.dataclass
class Command:
    """
    Base class of the evaluation commands.

    Parameters
    ----------
    dirs
        Directory layout of the analysis.

    Attributes
    ----------
    OUTPUT
        Name of the command-specific output subdirectory under
        ``dirs.evaluate``. Empty if the command does not use ``output``.
    """

    _: dc.KW_ONLY
    dirs: Dirs

    OUTPUT: ClassVar[str] = ''

    @functools.cached_property
    def output(self):
        """
        Command-specific output directory, created if missing.

        Returns
        -------
        pathlib.Path
            ``dirs.evaluate / OUTPUT``.

        Raises
        ------
        ValueError
            If ``OUTPUT`` is not set.
        """
        if not self.OUTPUT:
            raise ValueError

        output = self.dirs.evaluate / self.OUTPUT
        output.mkdir(exist_ok=True)

        return output

    def _scan(self, name: Literal['models', 'error']):
        """
        Scan an aggregated result table.

        Parameters
        ----------
        name
            ``'models'`` for the per-case optimization results
            (``01.models.parquet``) or ``'error'`` for the long-format
            parameter errors (``02.error.parquet``), both written by
            ``s01aggregate``.

        Returns
        -------
        polars.LazyFrame
        """
        n = {'models': '01.models', 'error': '02.error'}[name]
        return pl.scan_parquet(self.dirs.evaluate / f'{n}.parquet')


def typst_table(table: pl.DataFrame):
    """
    Format a summary table as CSV for a Typst table.

    Rows are sorted by resolution, beta (descending) and optimizer order;
    every cell is wrapped as Typst content (``[...]``), missing values are
    shown as ``--``, and an empty ``none`` column is appended.

    Parameters
    ----------
    table
        Summary table with ``dataset.resolution``, ``dataset.beta`` and
        ``optimizer`` columns.

    Returns
    -------
    str
        CSV text.
    """
    return (
        table
        .sort(
            [
                'dataset.resolution',
                'dataset.beta',
                pl.col('optimizer').replace_strict(OPTIMIZERS_ORDER),
            ],
            descending=[False, True, False],
        )
        .with_columns(
            pl.col('optimizer').replace({'Reference': 'Reference Model'}),
            pl.col('dataset.beta').replace_strict({
                '10%': '$beta_0.1$',
                '50%': '$beta_0.5$',
            }),
        )
        .with_columns(pl.format('[{}]', pl.all().fill_null('--')))
        .with_columns(pl.lit(None).alias('none'))
        .write_csv()
        .replace('"[""***""]"', '"***"')
    )


app = App(
    config=[
        Toml('env.toml', use_commands_as_keys=False),
        Toml('conf.toml', root_keys='evaluate'),
    ],
    result_action=['call_if_callable', 'print_non_int_sys_exit'],
)
