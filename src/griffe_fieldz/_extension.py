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

# ---------------------------------------------- WIP

from typing import Any, TypedDict
import fieldz
from griffe import (
    Class,
    DocstringSection,
    DocstringSectionKind,
    Extension,
    Kind,
    dynamic_import,
)
from rich import print
import sys

if "." not in sys.path:
    sys.path.append(".")

sources = {
    "docstring": {
        "attributes": {
            "attr": {
                "name": "attr",
                # "annotation": ExprName(name="int", parent=Class("MyClass", 4, 19)),
                "description": "Some attribute doc\n",
            }
        },
        "parameters": {},
    },
    "members": {
        "attr2": {
            "kind": "attribute",
            "name": "attr2",
            "lineno": 19,
            "endlineno": 19,
            "docstring": ...,
            "labels": {"class-attribute", "instance-attribute"},
            "members": {},
            "value": "1",
            # 'annotation': ExprName(name='int', parent=Class('MySub', 17, 20))
        }
    },
}


class AnnotationDict(TypedDict, total=False):
    name: str
    cls: str


class DocsSectionDict(TypedDict, total=False):
    name: str
    annotation: str | dict | None
    description: str
    value: str | None


class MembersDict(TypedDict, total=False):
    kind: Kind
    name: str
    lineno: int
    endlineno: int
    docstring: dict[str, Any]


class DocstringDict(TypedDict, total=False):
    value: str
    lineno: int
    endlineno: int


class FieldInfo(TypedDict, total=False):
    name: str
    type: Any
    description: str
    title: str
    default: Any
    default_factory: Any
    repr: bool
    hash: bool | None
    init: bool
    compare: bool
    metadata: dict
    kw_only: bool
    frozen: bool


class ClassInfo(TypedDict, total=False):
    docstring: dict[DocstringSectionKind, dict[str, DocsSectionDict]]
    members: dict[str, MembersDict]
    fields: dict[str, fieldz.Field]


class MergedFieldInfo(TypedDict, total=False):
    name: str
    field_info: DocsSectionDict
    doc_parameter: DocsSectionDict
    doc_attribute: DocsSectionDict
    doc_inline: str
    metadata_description: str


class MyExtension(Extension):
    def on_class_members(self, *, cls: Class, **kwargs):
        info = get_class_info(cls)
        print(info)
        print(merge_class_info(info))
        print(merge_field_info(merge_class_info(info)["attr"], {}))
        return super().on_class_members(cls=cls, **kwargs)


def get_class_info(cls: Class) -> ClassInfo:
    info: ClassInfo = {}
    if cls.docstring:
        info["docstring"] = dump_docs(cls.docstring.parsed)
    info["members"] = dump(cls.members)
    runtime_obj = dynamic_import(cls.path)
    info["fields"] = {f.name: f for f in fieldz.fields(runtime_obj)}
    return info


def _get_value(field: fieldz.Field) -> str:
    if field.default is fieldz.Field.MISSING:
        return "MISSING"
    return field.default


def merge_class_info(
    info: ClassInfo, config: dict | None = None
) -> dict[str, MergedFieldInfo]:
    config = config or {}
    out: dict[str, MergedFieldInfo] = {}
    docs = info.get("docstring", {})
    doc_params = docs.get(DocstringSectionKind.parameters, {})
    doc_attrs = docs.get(DocstringSectionKind.attributes, {})
    members = info.get("members", {})
    for name, field in info["fields"].items():
        member = members.get(name, {})
        out[name] = {
            "name": name,
            "field_info": {
                "value": _get_value(field),
                "annotation": str(field.type),
                "description": field.description or "",
            },
            "doc_parameter": doc_params.get(name, {}),
            "doc_attribute": doc_attrs.get(name, {}),
            "doc_inline": member.get("docstring", {}).get("value", ""),
            "metadata_description": field.metadata.get("description", ""),
        }

    return out


def merge_field_info(
    field_data: MergedFieldInfo, config: dict[str, Any] | None
) -> DocsSectionDict:
    """
    Merge all doc sources for a single field according to the provided config.

    :param field_data: The dictionary containing merged raw info from each doc source.
    :param config: A dictionary containing user preferences (see doc above).
    :return: The final doc string for this field.
    """
    config = config or {}

    # Filter out None or empty
    # Gather possible doc strings in a dictionary keyed by source
    doc_sources = {
        "doc_parameter": field_data.get("doc_parameter", {}).get("description"),
        "doc_attribute": field_data.get("doc_attribute", {}).get("description"),
        "doc_inline": field_data.get("doc_inline"),
        "metadata_description": field_data.get("metadata_description"),
    }

    # Filter out None or empty
    doc_sources = {k: v for k, v in doc_sources.items() if v}

    merge_strategy = config.get("doc_merge_strategy", "prefer_first")
    doc_priority = config.get(
        "doc_priority",
        ["metadata_description", "doc_inline", "doc_parameter", "doc_attribute"],
    )

    final_doc: DocsSectionDict = {}

    # If the user wants to combine them in some way:
    if merge_strategy == "concatenate":
        # Gather them in priority order, skipping missing ones
        non_empty_docs = []
        for source in doc_priority:
            doc_value = doc_sources.get(source)
            if doc_value:
                non_empty_docs.append(doc_value.strip())

        # Join them with the configured delimiter
        delimiter = config.get("doc_delimiter", "\n\n")
        final_doc["description"] = delimiter.join(non_empty_docs)

    elif merge_strategy == "prefer_first":
        # Just pick the first in priority order
        for source in doc_priority:
            doc_value = doc_sources.get(source)
            if doc_value:
                final_doc["description"] = doc_value.strip()
                break
    else:
        # Default fallback: same as "concatenate"
        final_doc["description"] = "\n\n".join(s for s in doc_sources.values() if s)

    # Optionally show default value (if present and user wants it)
    if config.get("show_default_values", True):
        default_val = field_data.get("field_info", {}).get("value")
        if default_val is not None:
            final_doc["description"] += f"\n\n**Default value:** `{default_val}`"

    return final_doc


def dump_docs(
    sections: list[DocstringSection],
) -> dict[DocstringSectionKind, dict[str, DocsSectionDict]]:
    out: dict[DocstringSectionKind, dict[str, DocsSectionDict]] = {}
    for section in sections:
        if isinstance(section.value, list) and all(
            hasattr(item, "name") for item in section.value
        ):
            out[section.kind] = {item.name: dump(item) for item in section.value}
        else:
            out[section.kind] = dump(section.value)
    return out


def dump(obj: object, **kwargs: Any) -> Any:
    # if isinstance(obj, DocstringParameter):
    #     return dump(obj)
    #     return {p.name: dump(p.value) for p in obj.value}
    if isinstance(obj, dict):
        return {k: dump(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [dump(v) for v in obj]
    if hasattr(obj, "as_dict"):
        return dump(obj.as_dict(**kwargs))
    return obj
