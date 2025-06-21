from __future__ import annotations
from dataclasses import field, dataclass
import re
import ast
from types import UnionType
from typing import Any, Type, Union, cast, get_origin, get_args

SUPPORTED_TYPES: dict[str, Type[Any]] = {
    "int": int,
    "list[int]": list[int],
    "float": float,
    "list[float]": list[float],
    "str": str,
    "list[str]": list[str],
}

class MiniDocStringType:
    """
    Represents a parsed and validated argument or return type.

    Supports primitive types (int, float, str), List types (list[int], list[str]), 
    and Optional types (Optional[int], Optional[list[int]]).

    Attributes:
        _type (Type[Any]):
            The Python type stored without the Optional[...] wrapper.

        _optional (bool):
            Indicates whether the type is optional, i.e., if it was passed as Optional[...].

    Raises:
        ValueError:
            If the provided type string is not supported.
    """

    _type: Type[Any]
    _optional: bool

    def __init__(self, type_input: str | Type[Any] | UnionType):
        """
        Initializes the MiniDocStringType from a type string or a Python type.

        Args:
            type_input (str | Type[Any] | UnionType):
                Either a type description string or a Python type.

        Raises:
            ValueError:
                If the type string is not recognized or the input is unsupported.
        """
        pytype_raw: Any

        if isinstance(type_input, str):
            type_str = type_input.strip()
            is_optional = False

            # Normalize "X | None" and "None | X" to Optional[X]
            if match := re.match(r"^(None\s*\|\s*(.*)|(.*)\s*\|\s*None)$", type_str):
                type_str = match.group(2) or match.group(3)
                type_str = type_str.strip()
                is_optional = True

            # Handle Optional[X] notation directly
            if not is_optional and (match := re.match(r"^Optional\[(.*)\]$", type_str)):
                type_str = match.group(1).strip()
                is_optional = True

            base_type = SUPPORTED_TYPES.get(type_str)
            if base_type is None:
                raise ValueError(f"Unsupported type string: {type_input!r}")

            pytype_raw = Union[base_type, None] if is_optional else base_type
        else:
            # Pylance infers that non-str inputs must be Type or UnionType from the hint.
            # But users can pass anything at runtime, so we still need this runtime check.
            if not isinstance(
                type_input, type | UnionType
            ):  # type: ignore[reportUnnecessaryIsInstance]
                raise ValueError(f"Unsupported type input: {type_input!r}")
            pytype_raw = type_input

        self._optional = False

        if get_origin(pytype_raw) in (Union, UnionType):
            args = get_args(pytype_raw)
            if len(args) == 2 and type(None) in args:
                self._optional = True
                self._type: Type[Any] = args[0] if args[1] is type(None) else args[1]
            else:
                self._type = cast(Type[Any], pytype_raw)
        else:
            self._type = cast(Type[Any], pytype_raw)

    def __eq__(self, other: object) -> bool:
        """
        Check whether two MiniDocStringType instances are equal.

        Args:
            other (object):
                The object to compare against.

        Returns:
            bool:
                True if both instances have the same type and optionality; False otherwise.
        """
        if not isinstance(other, MiniDocStringType):
            return NotImplemented
        return self._type == other._type and self._optional == other._optional

    def is_optional(self) -> bool:
        """
        Check if the argument or return value is optional (i.e., accepts None).

        Returns:
            bool:
                True if the argument or return value is optional, False otherwise.
        """
        return self._optional

    def is_list(self) -> MiniDocStringType | None:
        """
        Check if the argument or return value is a List and return its inner type.

        Returns:
            MiniDocStringType | None:
                The inner type of the List if the argument or return value is a List,
                otherwise None.
        """
        return MiniDocStringType(self._get_inner_type()) if self._is_list() else None

    def _is_list(self) -> bool:
        """
        Check if the underlying type is a List.

        Returns:
            bool:
                True if the type is a List, False otherwise.
        """
        return self._type is list or get_origin(self._type) is list

    def _get_inner_type(self) -> Type[Any]:
        """
        Get the inner type if this is a list[X].

        Returns:
            Type[Any]:
                The inner type if applicable, otherwise the type itself.
        """
        origin = get_origin(self._type)
        if origin is list:
            args = get_args(self._type)
            if args:
                return args[0]
        return self._type

    def matches_by_value(self, obj: Any) -> bool:
        """
        Check if a given object matches the expected type.

        Args:
            obj (Any):
                The object to validate.

        Returns:
            bool:
                True if the object matches the expected type, False otherwise.
        """
        if obj is None:
            return self.is_optional()

        if self._is_list():
            if not isinstance(obj, list):
                return False
            inner_type = self._get_inner_type()
            return all(isinstance(elem, inner_type) for elem in cast(list[Any], obj))

        return isinstance(obj, self._type)

    def matches_by_type(self, t: Type[Any]) -> bool:
        """
        Check if a given type matches the expected type.

        Args:
            t (Type[Any]):
                The type to validate.

        Returns:
            bool:
                True if the type matches the expected type, False otherwise.
        """
        if t is type(None):
            return self.is_optional()

        if self._is_list():
            return t is list

        return t is self._type

    def __repr__(self) -> str:
        """
        Return a readable string representation of the MiniDocStringType.

        Returns:
            str:
                The string representation.
        """
        return f"ArgType({self.to_string()})"

    def to_string(self, strip_optional: bool = False) -> str:
        """
        Dynamically reconstructs the type string.

        Args:
            strip_optional (bool):
                If True, the returned type string will omit the Optional[...] wrapper
                and show only the inner type. This is useful when the optionality is
                expressed separately (e.g., in a schema with an 'optional' flag).

        Returns:
            str:
                The human-readable reconstructed type string.
        """
        type_str = ""

        if self._is_list():
            inner = self._get_inner_type()
            inner_name = inner.__name__ if hasattr(inner, '__name__') else str(inner)
            type_str = f"list[{inner_name}]"
        else:
            type_str = self._type.__name__ if hasattr(self._type, '__name__') else str(self._type)

        if self.is_optional() and not strip_optional:
            type_str = f"Optional[{type_str}]"

        return type_str
    def cast(self, value: str | int | list[Any] | None, allow_scalar_to_list: bool = False) -> Any:
        """
        Attempt to cast a value (str, int, list, or None) to the target type.

        Args:
            value (str | int | list[Any] | None):
                The input value. May be a string, integer, list, or None.

            allow_scalar_to_list (bool):
                If True, a single scalar value will be promoted to a one-element list 
                if the target type expects a list.

        Returns:
            Any:
                The value cast to the expected type.

        Raises:
            ValueError:
                If the cast fails or the type is unsupported.
        """
        if value is None:
            if self.is_optional():
                return None
            raise ValueError(f"Cannot cast None to {self.to_string()}")

        if self._is_list():
            if isinstance(value, str):
                try:
                    parsed = ast.literal_eval(value)
                except Exception as e:
                    raise ValueError(
                        f"Failed to cast list value '{value}' to {self.to_string()}"
                    ) from e
            else:
                parsed = value

            if not isinstance(parsed, list):
                if allow_scalar_to_list:
                    parsed = [parsed]
                else:
                    raise ValueError(f"Expected a list, got {type(parsed).__name__}")

            inner_type = self._get_inner_type()
            try:
                return [inner_type(elem) for elem in cast(list[Any], parsed)]
            except Exception as e:
                raise ValueError(
                    f"Failed to cast elements of {parsed!r} to {self.to_string()}"
                ) from e
        else:
            try:
                return self._type(value)
            except Exception as e:
                raise ValueError(
                    f"Failed to cast value '{value}' to {self.to_string()}"
                ) from e


@dataclass
class MiniDocStringArg:
    """
    Represents a single argument in a Google-style Args: section.

    Attributes:
        name (str):
            The name of the parameter.

        pytype (MiniDocStringType):
            The type hint provided in parentheses.

        desc (str):
            The full description (supports multiline).
    """
    name: str
    pytype: MiniDocStringType
    desc: str


@dataclass
class MiniDocString:
    """
    Parses a Google-style docstring into structured components.

    This mini parser extracts and stores structured information from a function's
    docstring, including descriptions, parameter types, return types, and exceptions.

    Attributes:
        _arg_name_to_object (dict[str, MiniDocStringArg]):
            Internal mapping of argument names to their parsed representations.

        description_short (str):
            The first paragraph of the docstring, typically a one-line summary.

        description_detailed (str):
            Any additional description following the short summary, up to the
            parameter or return sections.

        args (list[MiniDocStringArg]):
            A list of parsed arguments from the Args: section.

        return_type (MiniDocStringType | None):
            The parsed return type, if specified.

        return_desc (str | None):
            The description of the return value, if present.

        raises (str | None):
            The description of raised exceptions from the Raises: section.
    """
    _arg_name_to_object: dict[str, MiniDocStringArg]
    description_short: str = ""
    description_detailed: str = ""
    args: list[MiniDocStringArg] = field(default_factory=list[MiniDocStringArg])
    return_type: MiniDocStringType | None = None
    return_desc: str | None = None
    raises: str | None = None

    def __init__(self, docstring: str | None):
        """
        Initialize and parse a Google-style docstring into structured sections.

        Args:
            docstring (str | None):
                The raw docstring to parse. If None or empty, the parser initializes
                with empty descriptions and no arguments, return type, or exceptions.
        """
        self.description_short, self.description_detailed = "", ""
        self.args, self.return_type, self.return_desc, self.raises = [], None, None, None

        if not docstring:
            return

        doc = docstring.strip("\n")

        # 1. Parse description section (up to Args/Returns/Raises)
        g = re.search(r"(.*?)(?=^\s*(Args:|Returns:|Raises:))", doc, flags=re.MULTILINE | re.DOTALL)
        if g:
            lines = g[1].splitlines()
            doc = doc[g.end():]  # remove what we've already parsed
        else:
            lines = doc.splitlines()
            doc = ""

        if not lines:
            raise ValueError("Missing description before Args/Returns/Raises block")

        main_indent = cast(re.Match[str], re.match(r"^(\s*)", lines[0]))[1]
        lines = [line[len(main_indent):].rstrip() for line in lines]

        try:
            i = lines.index("")
            self.description_short = "\n".join(lines[:i])
            self.description_detailed = "\n".join(lines[i+1:]).rstrip("\n")
        except ValueError:
            self.description_short = "\n".join(lines)

        # 2. Parse Args section if present
        g = re.search(r"Args:(.*?)(?=\n\s*(Args:|Returns:|Raises:)|$)", doc, flags=re.DOTALL)
        if g:
            args_block = g[1].strip("\n")
            lines = args_block.splitlines()
            tabulator = cast(re.Match[str], re.match(r"^(\s*)", lines[0]))[1]
            lines = [line[len(tabulator):].rstrip() for line in lines]

            current: MiniDocStringArg | None = None
            buffer: list[str] = []

            for line in lines:
                match = re.match(r"\s*(\w+)\s*\(([^)]+)\):\s*(.*)?", line)
                if match:
                    if current is not None:
                        current.desc = '\n'.join(buffer).strip("\n")
                        self.args.append(current)
                    current = MiniDocStringArg(match[1], MiniDocStringType(match[2]), "")
                    buffer = [] if not match[3] else [match[3]]
                else:
                    assert current is not None
                    buffer.append(line[len(tabulator) - len(main_indent):])

            if current is not None:
                current.desc = '\n'.join(buffer).strip("\n")
                self.args.append(current)

        # 3. Parse Returns section if present
        g = re.search(r"Returns:(.*?)(?=\n\s*(Args:|Returns:|Raises:)|$)", doc, flags=re.DOTALL)
        if g:
            lines = g[1].strip("\n").splitlines()
            tabulator = cast(re.Match[str], re.match(r"^(\s*)", lines[0]))[1]
            lines = [line[len(tabulator):].rstrip() for line in lines]

            if lines and (match := re.match(r"^\s*(\w+)\s*:\s*(.*)?", lines[0])):
                self.return_type = MiniDocStringType(match[1])
                return_desc = match[2] or ""

                if len(lines) > 1:
                    lines = lines[1:]
                    tabulator = cast(re.Match[str], re.match(r"^(\s*)", lines[0]))[1]
                    lines = [line[len(tabulator):].rstrip() for line in lines]
                    if return_desc != "":
                        return_desc += " " + " ".join(lines).strip()
                    else:
                        return_desc = " ".join(lines).strip()

                self.return_desc = return_desc
            else:
                self.return_type = None
                self.return_desc = ' '.join(line.strip() for line in lines).strip()

        # 4. Parse Raises section if present
        g = re.search(r"Raises:(.*?)(?=\n\s*(Args:|Returns:|Raises:)|$)", doc, flags=re.DOTALL)
        if g:
            lines = g[1].strip("\n").splitlines()
            tabulator = cast(re.Match[str], re.match(r"^(\s*)", lines[0]))[1]
            lines = [line[len(tabulator):].rstrip() for line in lines]
            self.raises = ' '.join(line.strip() for line in lines).strip()

        self._arg_name_to_object = {a.name: a for a in self.args}

    def get_arg_by_name(self, name: str) -> MiniDocStringArg:
        """
        Retrieve the parsed argument definition by its name.

        Args:
            name (str):
                The name of the argument to retrieve.

        Returns:
            MiniDocStringArg:
                The parsed argument object corresponding to the given name.

        Raises:
            KeyError:
                If the argument name does not exist in the parsed docstring.
        """
        return self._arg_name_to_object[name]

    def validate_runtime_args(self, runtime_args: dict[str, Any]) -> None:
        """
        Validate runtime arguments against the parsed docstring definition.

        Checks that:
          - All required arguments are present.
          - No unexpected arguments are included.
          - Each argument matches the expected type (e.g., int, float, str, list).

        Args:
            runtime_args (dict[str, Any]):
                A dictionary of argument names to their runtime values.

        Raises:
            ValueError:
                If an argument is missing, has an unexpected name, or does not match
                the expected type.
        """
        expected_names = {arg.name for arg in self.args}
        provided_names = set(runtime_args.keys())

        # Check for unexpected arguments
        if unexpected := provided_names - expected_names:
            raise ValueError(f"Unexpected arguments: {', '.join(unexpected)}")

        # Check for missing arguments
        if missing := expected_names - provided_names:
            raise ValueError(f"Missing required arguments: {', '.join(missing)}")

        # Validate type
        for arg in self.args:
            value = runtime_args[arg.name]
            arg_type = arg.pytype

            if not arg_type.matches_by_value(value):
                raise ValueError(
                    f"Argument '{arg.name}' expected {arg_type}, but got {type(value).__name__}"
                )

    def to_schema_yaml(self, top_level_key: str, function_name: str) -> str:
        """
        Serializes this MiniDocString into a general-purpose YAML schema format.

        Args:
            top_level_key (str):
                Top-level key of the YAML schema (e.g., "intent", "action", "function")

            function_name (str):
                The name of the function or unit of logic (used as the value of the top-level key).

        Returns:
            str:
                A valid YAML block with fields:
                    - <top_level_key>: <function_name>
                    - description (brief + detailed)
                    - parameters (each with name, type, description, optional)
                    - return (if present, with type and description)
        """
        def _flatten(text: str) -> str:
            return " ".join(line.strip() for line in text.strip().splitlines())

        lines = [f"- {top_level_key}: {function_name}"]

        # Combine and flatten the description
        full_description = _flatten(self.description_short)
        if self.description_detailed:
            full_description += " " + _flatten(self.description_detailed)
        lines.append(f"  description: {full_description}")

        if not self.args:
            lines.append("  parameters: []")
        else:
            lines.append("  parameters:")
            for arg in self.args:
                lines.append(f"    - name: {arg.name}")
                lines.append(f"      type: {arg.pytype.to_string(strip_optional=True)}")
                lines.append(f"      description: {_flatten(arg.desc)}")
                lines.append(f"      optional: {arg.pytype.is_optional()}")

        if self.return_type and self.return_desc:
            lines.append("  return:")
            lines.append(f"    type: {self.return_type.to_string()}")
            lines.append(f"    description: {_flatten(self.return_desc)}")

        return "\n".join(lines)
