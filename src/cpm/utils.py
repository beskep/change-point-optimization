"""Shared utilities."""

import itertools
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Literal, TypeVar, overload

import attrs
import cyclopts
import matplotlib as mpl
from cyclopts.core import _result_action_converter  # ruff:ignore[import-private-name]
from matplotlib import ticker

if TYPE_CHECKING:
    from collections.abc import Iterable

    from cyclopts._result_action import ResultAction, ResultActionSingle
    from matplotlib.axes import Axes


__all__ = ['App', 'equal_scale']

F = TypeVar('F', bound=Callable)


def _is_helper(name: str | Iterable[str] | None) -> bool:
    """
    Return whether ``name`` refers to a built-in help or version command.

    Parameters
    ----------
    name
        Command name(s) passed to ``App.command``.

    Returns
    -------
    bool
        True if any of the names is ``-h``, ``--help`` or ``--version``.
    """
    if name is None:
        return False

    names = [name] if isinstance(name, str) else name
    return any(n in {'-h', '--help', '--version'} for n in names)


@attrs.define
class App(cyclopts.App):
    """
    ``cyclopts.App`` that lists commands in registration order.

    Compared with ``cyclopts.App``, this class

    - assigns an increasing ``sort_key`` to each registered command, so that the
      help page lists commands in the order they are defined;
    - strips the application name, if present, as a prefix from command
      names before applying the default name transform;
    - defaults ``result_action`` to ``('call_if_callable',
      'print_non_int_sys_exit')`` so that a command registered as a dataclass
      is instantiated and then called.
    """

    _count: itertools.count = attrs.field(factory=itertools.count)
    result_action: ResultAction | ResultActionSingle | None = attrs.field(
        default=('call_if_callable', 'print_non_int_sys_exit'),
        converter=_result_action_converter,
        kw_only=True,
    )

    def _remove_prefix(self, s: str) -> str:
        """
        Transform a Python name into a command name.

        Parameters
        ----------
        s
            Python name of the registered object.

        Returns
        -------
        str
            ``s`` with the (first) application name removed as a prefix,
            transformed by ``cyclopts.default_name_transform``. Falls back to
            the transformed ``s`` if the result would be empty.
        """
        t = s.removeprefix(self.name[0]) if self.name else s
        t = cyclopts.default_name_transform(t)
        return t or cyclopts.default_name_transform(s)

    @overload
    def command(
        self,
        obj: F,
        name: str | Iterable[str] | None = ...,
        sort_key: Any = ...,
        name_transform: Callable[[str], str] | None = ...,
        **kwargs: Any,
    ) -> F: ...

    @overload
    def command(
        self,
        obj: None = None,
        name: str | Iterable[str] | None = ...,
        sort_key: Any = ...,
        name_transform: Callable[[str], str] | None = ...,
        **kwargs: Any,
    ) -> Callable[[F], F]: ...

    def command(
        self,
        obj: Callable | None = None,
        name: str | Iterable[str] | None = None,
        sort_key: Any = None,
        name_transform: Callable[[str], str] | None = None,
        **kwargs: Any,
    ):  # ty:ignore[invalid-method-override]
        """
        Register a command; see ``cyclopts.App.command``.

        Parameters
        ----------
        obj
            Function, class or sub-application to register. If None, a
            decorator is returned.
        name
            Command name(s). If None, the transformed name of ``obj`` is used.
        sort_key
            Sort key on the help page. If None, the registration order is
            used; help/version commands are left unsorted.
        name_transform
            Function converting the Python name into the command name.
            If None, ``_remove_prefix`` is used. Ignored when ``obj`` is a
            ``cyclopts.App``.
        **kwargs
            Passed to ``cyclopts.App.command``.

        Returns
        -------
        callable
            ``obj`` itself, or a decorator if ``obj`` is None.
        """
        if _is_helper(name):
            sort_key = None
        elif sort_key is None:
            sort_key = next(self._count)

        if isinstance(obj, cyclopts.App):
            obj.sort_key = sort_key
        else:
            if name_transform is None:
                name_transform = self._remove_prefix

            kwargs['sort_key'] = sort_key
            kwargs['name_transform'] = name_transform

        return super().command(obj=obj, name=name, **kwargs)


def equal_scale(
    ax: Axes,
    *,
    identity_line: bool | Mapping = True,
    locator: Literal['auto'] | ticker.Locator | None = 'auto',
) -> None:
    """
    Give both axes of ``ax`` the same scale and data range.

    The data limits are made symmetric in x and y, the aspect ratio is set to
    1, and optionally an identity line (``y = x``) is drawn.

    Parameters
    ----------
    ax
        Target axes.
    identity_line
        Whether to draw the identity line. If a mapping, it is passed as
        keyword arguments to ``Axes.axline``; if True, the current grid style
        (``grid.*`` rcParams) is used.
    locator
        Major tick locator shared by both axes. ``'auto'`` uses
        ``matplotlib.ticker.AutoLocator``; None leaves the tick locators
        unchanged.
    """
    ax.set_aspect('equal', adjustable='box')

    ax.dataLim.update_from_data_xy(ax.dataLim.get_points()[:, ::-1], ignore=False)
    ax.autoscale_view()

    if identity_line:
        if isinstance(identity_line, Mapping):
            style = identity_line
        else:
            keys = ('grid.color', 'grid.linestyle', 'grid.linewidth', 'grid.alpha')
            style = {k.removeprefix('grid.'): mpl.rcParams[k] for k in keys}

        p = ax.dataLim.get_points()[:, 0].mean()
        ax.axline((p, p), slope=1, **style)

    if locator:
        loc = ticker.AutoLocator() if locator == 'auto' else locator
        ax.xaxis.set_major_locator(loc)
        ax.yaxis.set_major_locator(loc)
        # ignore minor locator
