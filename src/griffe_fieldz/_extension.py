"""Griffe Fieldz extension."""

from __future__ import annotations

import inspect
import textwrap
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Iterable, Sequence

import fieldz
from fieldz._repr import display_as_type
from griffe import (
    Class,
    Docstring,
    DocstringAttribute,
    DocstringParameter,
    DocstringSectionAttributes,
    DocstringSectionParameters,
    Extension,
    Object,
    ObjectNode,
    dynamic_import,
    get_logger,
    parse_docstring_annotation,
)

if TYPE_CHECKING:
    import ast

    from griffe import Expr, Inspector, Visitor

logger = get_logger(__name__)


class FieldzExtension(Extension):
    """Griffe extension that injects field information for dataclass-likes."""

    def __init__(
        self,
        object_paths: list[str] | None = None,
        include_private: bool = False,
        include_inherited: bool = False,
        **kwargs: Any,
    ) -> None:
        self.object_paths = object_paths
        self._kwargs = kwargs
        self.include_private = include_private
        self.include_inherited = include_inherited

    def on_class_members(
        self,
        *,
        node: ast.AST | ObjectNode,
        cls: Class,
        agent: Visitor | Inspector,
        **kwargs: Any,
    ) -> None:
        if isinstance(node, ObjectNode):
            return  # skip runtime objects
        if self.object_paths and cls.path not in self.object_paths:
            return  # skip objects that were not selected

        # import object to get its evaluated docstring
        try:
            runtime_obj = dynamic_import(cls.path)
        except ImportError:
            logger.debug(f"Could not get dynamic docstring for {cls.path}")
            return

        try:
            fieldz.get_adapter(runtime_obj)
        except TypeError:
            return
        self._inject_fields(cls, runtime_obj)

    # ------------------------------

    def _inject_fields(self, obj: Object, runtime_obj: Any) -> None:
        # update the object instance with the evaluated docstring
        docstring = inspect.cleandoc(getattr(runtime_obj, "__doc__", "") or "")
        if not obj.docstring:
            obj.docstring = Docstring(docstring, parent=obj)
        sections = obj.docstring.parsed

        # collect field info
        fields = fieldz.fields(runtime_obj)
        if not self.include_inherited:
            annotations = getattr(runtime_obj, "__annotations__", {})
            fields = tuple(f for f in fields if f.name in annotations)
        from rich import print

        params, attrs = _fields_to_params(fields, obj.docstring, self.include_private)
        if runtime_obj.__name__ == "PosteriorStandardDeviation":
            breakpoint()
        # merge/add field info to docstring
        if params:
            for x in sections:
                if isinstance(x, DocstringSectionParameters):
                    _merge(x, params)
                    break
            else:
                sections.insert(1, DocstringSectionParameters(params))
        if attrs:
            for x in sections:
                if isinstance(x, DocstringSectionAttributes):
                    _merge(x, params)
                    break
            else:
                sections.append(DocstringSectionAttributes(attrs))


def _to_annotation(type_: Any, docstring: Docstring) -> str | Expr | None:
    """Create griffe annotation for a type."""
    if type_:
        return parse_docstring_annotation(
            display_as_type(type_, modern_union=True), docstring
        )
    return None


def _default_repr(field: fieldz.Field) -> str | None:
    """Return a repr for a field default."""
    try:
        if field.default is not field.MISSING:
            return repr(field.default)
        if (factory := field.default_factory) is not field.MISSING:
            try:
                sig = inspect.signature(factory)
            except ValueError:
                return repr(factory)
            else:
                if len(sig.parameters) == 0:
                    with suppress(Exception):
                        return repr(factory())

            return "<dynamic>"
    except Exception as exc:
        logger.warning("Failed to get default repr for %s: %s", field.name, exc)
        pass
    return None


def _fields_to_params(
    fields: Iterable[fieldz.Field],
    docstring: Docstring,
    include_private: bool = False,
) -> tuple[list[DocstringParameter], list[DocstringAttribute]]:
    """Get all docstring attributes and parameters for fields."""
    params: list[DocstringParameter] = []
    attrs: list[DocstringAttribute] = []
    for field in fields:
        try:
            desc = field.description or field.metadata.get("description", "") or ""
            if not desc and (doc := getattr(field.default_factory, "__doc__", None)):
                desc = inspect.cleandoc(doc) or ""

            kwargs: dict = {
                "name": field.name,
                "annotation": _to_annotation(field.type, docstring),
                "description": textwrap.dedent(desc).strip(),
                "value": _default_repr(field),
            }

            if field.init:
                params.append(DocstringParameter(**kwargs))
            elif include_private or not field.name.startswith("_"):
                attrs.append(DocstringAttribute(**kwargs))
        except Exception as exc:
            logger.warning("Failed to parse field %s: %s", field.name, exc)

    return params, attrs


def _merge(
    existing_section: DocstringSectionParameters | DocstringSectionAttributes,
    field_params: Sequence[DocstringParameter],
) -> None:
    """Update DocstringSection with field params (if missing)."""
    existing_members = {x.name: x for x in existing_section.value}

    for param in field_params:
        if existing := existing_members.get(param.name):
            # if the field already exists ...
            # extend missing attributes with the values from the fieldz params
            if existing.value is None and param.value is not None:
                existing.value = param.value
            if existing.description is None and param.description:
                existing.description = param.description
            if existing.annotation is None and param.annotation is not None:
                existing.annotation = param.annotation
        else:
            # otherwise, add the missing fields
            existing_section.value.append(param)  # type: ignore
