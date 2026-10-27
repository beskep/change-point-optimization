"""
Directory layout of the analysis workspace.

The root directory is given in ``env.toml``; every other directory is
declared relative to its parent and resolved to an absolute path on
initialisation.
"""

import dataclasses as dc
from pathlib import Path


@dc.dataclass
class BaseDirs:
    """
    Base class for a tree of directories relative to a common root.

    Every field other than ``root`` must be either a ``pathlib.Path``, interpreted
    relative to ``root``, or a nested ``BaseDirs``, whose own ``root`` is interpreted
    relative to this ``root``. Relative fields are made absolute in
    ``__post_init__``, but only when ``root`` itself is absolute.

    Parameters
    ----------
    root
        Root directory of the tree.

    Raises
    ------
    TypeError
        If a field is neither a ``pathlib.Path`` nor a ``BaseDirs``.
    """

    root: Path

    def __post_init__(self):
        if not self.root.is_absolute():
            return

        for field in (x.name for x in dc.fields(self)):
            if field == 'root':
                continue

            match value := getattr(self, field):
                case Path():
                    setattr(self, field, self.root / value)
                case BaseDirs():
                    value.root = self.root / value.root
                    value.__post_init__()
                case _:
                    raise TypeError(field, value)


@dc.dataclass
class Distribution(BaseDirs):
    """
    Output directories of the distribution analysis (``s01distribution``).

    Parameters
    ----------
    root
        Root directory, relative to ``Dirs.root``.
    raw
        Descriptive statistics, figures and change-point distribution of the
        measured CPMs.
    baseline
        Fitted baseline energy use distributions.

    Notes
    -----
    The parameters used to generate the synthetic datasets are written to
    ``data/`` in the repository, not to these directories.
    """

    root: Path = Path('02.dist')

    raw: Path = Path('01.raw')
    baseline: Path = Path('02.baseline')


@dc.dataclass
class Dirs(BaseDirs):
    """
    Top-level directory layout of the analysis.

    ``root`` is resolved to an absolute path and must exist.

    Parameters
    ----------
    root
        Root directory of the workspace (set in ``env.toml``).
    raw
        Raw input data (institution list, CPM models, AMI, ASOS weather).
    data
        Preprocessed data (``s00preprocess``).
    dist
        Distribution analysis outputs (``s01distribution``).
    optimize
        Optimization results of the Monte Carlo experiment (``s02optimize``).
    evaluate
        Evaluation outputs (``s03evaluate``).

    Raises
    ------
    FileNotFoundError
        If ``root`` does not exist.
    """

    raw: Path = Path('00.raw')
    data: Path = Path('01.data')
    dist: Distribution = dc.field(default_factory=Distribution)
    optimize: Path = Path('03.optimize')
    evaluate: Path = Path('04.evaluate')

    def __post_init__(self):
        self.root = self.root.resolve()
        self.root.stat()
        return super().__post_init__()


if __name__ == '__main__':
    import cyclopts
    import rich

    app = cyclopts.App(config=cyclopts.config.Toml('env.toml'))

    @app.default
    def config(dirs: Dirs):
        """Print the resolved directory layout."""
        rich.print(dirs)

    app()
