"""Generate sturdystats entity classes from sdk_config.json + openapi.json."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
CONFIG_PATH = ROOT / "sdk_config.json"
OPENAPI_PATH = ROOT / "openapi.json"
OUT_DIR = ROOT / "sturdystats"

OPENAPI_PREFIX = "/api/v1/orgs/{org-id}"


def load_openapi():
    if not OPENAPI_PATH.exists():
        print("openapi.json not found, fetching from server...")
        import urllib.request
        with urllib.request.urlopen("http://localhost:3333/openapi.json") as r:
            data = r.read()
        OPENAPI_PATH.write_bytes(data)
        print(f"  saved to {OPENAPI_PATH}")
    return json.loads(OPENAPI_PATH.read_text())


def openapi_path_key(short_route: str) -> tuple[str, str]:
    method, path = short_route.split(" ", 1)
    full = OPENAPI_PREFIX + path.replace("{dataset_id}", "{dataset-id}") \
                                .replace("{index_id}", "{index-id}") \
                                .replace("{clf_base_id}", "{clf-base-id}") \
                                .replace("{clf_model_id}", "{clf-model-id}")
    return method.lower(), full


def get_operation(spec: dict, short_route: str) -> dict:
    method, full_path = openapi_path_key(short_route)
    op = spec["paths"].get(full_path, {}).get(method)
    if op is None:
        raise KeyError(f"Route not found in OpenAPI: {method.upper()} {full_path}")
    return op


OPENAPI_TYPE_MAP = {
    "string": "str",
    "integer": "int",
    "number": "float",
    "boolean": "bool",
    "object": "dict",
    "array": "list",
}


def py_type(info: dict) -> str:
    """Map an OpenAPI property schema to a Python type annotation string."""
    enum = info.get("enum")
    if enum and all(isinstance(e, (str, int, bool)) for e in enum):
        return "Literal[" + ", ".join(repr(e) for e in enum) + "]"
    # nullable / union expressed via oneOf/anyOf: pick non-null branches
    for key in ("oneOf", "anyOf"):
        if key in info:
            branches = info[key]
            nullable = any(b.get("type") == "null" for b in branches)
            non_null = [b for b in branches if b.get("type") != "null"]
            inner = py_type(non_null[0]) if non_null else "Any"
            return f"Optional[{inner}]" if nullable else inner
    return OPENAPI_TYPE_MAP.get(info.get("type"), "Any")


def _unwrap_schema(schema: dict) -> dict:
    """Flatten a top-level allOf wrapper down to the object that has properties."""
    if "allOf" in schema and "properties" not in schema:
        for s in schema["allOf"]:
            if "properties" in s:
                return s
    return schema


def get_body_params(op: dict) -> list[dict]:
    rb = op.get("requestBody", {})
    schema = None
    for ct in ("application/json", "multipart/form-data"):
        if ct in rb.get("content", {}):
            schema = rb["content"][ct].get("schema", {})
            break
    if schema is None:
        return []
    schema = _unwrap_schema(schema)
    props = schema.get("properties", {})
    required = set(schema.get("required", []))
    params = []
    for name, info in props.items():
        params.append({
            "name": name,
            "py_name": name.replace("-", "_"),
            "required": name in required and info.get("default") is None,
            "description": info.get("description", ""),
            "default": info.get("default"),
            "py_type": py_type(info),
        })
    return params


def get_response_props(op: dict) -> list[dict]:
    """Return the property list of the lowest 2xx JSON success response, if any."""
    responses = op.get("responses", {})
    codes = sorted(c for c in responses if c.startswith("2"))
    for code in codes:
        content = responses[code].get("content", {})
        schema = content.get("application/json", {}).get("schema")
        if not schema:
            continue
        schema = _unwrap_schema(schema)
        props = schema.get("properties", {})
        if not props:
            continue
        out = []
        for name, info in props.items():
            out.append({
                "name": name,
                "py_type": py_type(info),
                "description": info.get("description", ""),
            })
        return out
    return []


def get_dataframe_columns(op: dict, response_key: list) -> list[dict]:
    """For a response_key path (e.g. ['datasets']) pointing at an array, return the
    item property list — these become the DataFrame columns.
    """
    responses = op.get("responses", {})
    codes = sorted(c for c in responses if c.startswith("2"))
    for code in codes:
        schema = responses[code].get("content", {}).get("application/json", {}).get("schema")
        if not schema:
            continue
        node = _unwrap_schema(schema)
        for k in (response_key or []):
            node = _unwrap_schema(node.get("properties", {}).get(k, {}))
        items = node.get("items")
        if not items:
            continue
        items = _unwrap_schema(items)
        props = items.get("properties", {})
        return [{
            "name": name,
            "py_type": py_type(info),
            "description": info.get("description", ""),
        } for name, info in props.items()]
    return []


def format_desc(desc: str, first_prefix: str, cont_indent: str) -> list[str]:
    """Render a possibly multi-line description as docstring lines.

    The first source line follows `first_prefix`; every continuation line is
    stripped and re-indented to `cont_indent` so wrapped text hangs neatly under
    the entry. Blank lines are preserved (without trailing whitespace).
    """
    parts = (desc or "").split("\n")
    out = [first_prefix + parts[0].strip()]
    for ln in parts[1:]:
        s = ln.strip()
        out.append(cont_indent + s if s else "")
    return out


def field_doc_lines(props: list[dict]) -> list[str]:
    """Render response/column field entries as aligned docstring lines."""
    lines = []
    for p in props:
        head = f"                {p['name']} ({p['py_type']}): "
        if p["description"]:
            lines.extend(format_desc(p["description"], head, " " * 20))
        else:
            lines.append(head.rstrip())
    return lines


def get_description(op: dict) -> str:
    return (op.get("description", "") or op.get("summary", "") or "")


def get_tag_description(spec: dict, tag_name: str) -> str:
    for t in spec.get("tags", []):
        if t["name"].lower() == tag_name.lower():
            return t.get("description", "")
    return ""


def render_method(method_cfg: dict, spec: dict, entity_cfg: dict, typedefs: list) -> str:
    name = method_cfg["name"]
    route = method_cfg["route"]
    http_method, path = route.split(" ", 1)
    http_method = http_method.upper()
    mtype = method_cfg.get("type")
    response_key = method_cfg.get("response_key")
    transform_import = method_cfg.get("transform")
    id_param = entity_cfg.get("id_param")
    id_field = entity_cfg.get("id_field")

    op = get_operation(spec, route)
    description = get_description(op)
    body_params = get_body_params(op) if http_method == "POST" else []
    pparams = re.findall(r"\{(\w+)\}", path)
    instance_params = [p for p in pparams if p != "org_id" and p != id_param]

    is_create = (name == "create")
    lines = []

    # --- signature ---
    if is_create:
        sig_parts = ["cls"]
    else:
        sig_parts = ["self"] + [f"{p}: str" for p in instance_params]

    if mtype == "upload_parquet":
        sig_parts.append("filepath: str")
    elif http_method == "POST":
        required_params = [bp for bp in body_params if bp["required"]]
        optional_params = [bp for bp in body_params if not bp["required"]]
        for bp in required_params:
            sig_parts.append(f"{bp['py_name']}: {bp['py_type']}")
        for bp in optional_params:
            d = bp["default"]
            t = bp["py_type"]
            if d is not None:
                sig_parts.append(f"{bp['py_name']}: {t} = {d!r}")
            elif t.startswith("Optional[") or t == "Any":
                sig_parts.append(f"{bp['py_name']}: {t} = None")
            else:
                sig_parts.append(f"{bp['py_name']}: Optional[{t}] = None")

    if is_create:
        sig_parts += ["org_id = None", "api_key = None", "base_url = None"]

    if response_key or mtype == "parquet_response":
        sig_parts.append("transform = None")

    # --- return type + optional TypedDict ---
    class_name = entity_cfg["class"]
    returns_doc = []
    if is_create:
        ret_ann = f'"{class_name}"'
        returns_doc = [f"            {class_name} instance (with .id set)."]
    elif mtype == "upload_parquet":
        ret_ann = "list[dict]"
        rprops = get_response_props(op)
        returns_doc = ["            list of response dicts, one per uploaded file, each with:"]
        returns_doc += field_doc_lines(rprops)
    elif mtype == "parquet_response":
        ret_ann = '"pd.DataFrame"'
        returns_doc = ["            pandas.DataFrame of the query result."]
    elif response_key:
        ret_ann = '"pd.DataFrame"'
        cols = get_dataframe_columns(op, response_key)
        returns_doc = ["            pandas.DataFrame with columns:"]
        returns_doc += field_doc_lines(cols)
    else:
        rprops = get_response_props(op)
        if rprops:
            td_name = class_name + "".join(w.capitalize() for w in name.split("_")) + "Response"
            td_lines = [f'{td_name} = TypedDict("{td_name}", {{']
            for p in rprops:
                td_lines.append(f'    {p["name"]!r}: {p["py_type"]},')
            td_lines.append("})")
            typedefs.append("\n".join(td_lines))
            ret_ann = td_name
            returns_doc = [f"            {td_name}, a dict with:"]
            returns_doc += field_doc_lines(rprops)
        else:
            ret_ann = "dict"

    if is_create:
        lines.append("    @classmethod")
    lines.append(f"    def {name}({', '.join(sig_parts)}) -> {ret_ann}:")

    # --- docstring ---  (order: description, route, params, returns)
    doc = ['        """']
    if description:
        doc.append(f"        {description}")
        doc.append("        ")
    doc.append(f"        Route: {http_method} {path}")

    # parameter docs
    arg_lines = []
    if mtype == "upload_parquet":
        arg_lines.append("            filepath: path to a .parquet file or directory of .parquet files")
    else:
        for bp in body_params:
            req_note = "" if bp["required"] else f" (default: {bp['default']!r})"
            head = f"            {bp['py_name']}{req_note}"
            if bp["description"]:
                arg_lines.extend(format_desc(bp["description"], head + " — ", " " * 16))
            else:
                arg_lines.append(head)
    if arg_lines:
        doc.append("        ")
        doc.append("        Args:")
        doc.extend(arg_lines)

    if returns_doc:
        doc.append("        ")
        doc.append("        Returns:")
        doc.extend(returns_doc)

    if transform_import:
        doc.append(f"        Default transform: {transform_import}")
    doc.append('        """')
    lines.extend(doc)

    # --- transform default ---
    if transform_import:
        mod, attr = transform_import.rsplit(".", 1)
        lines.append(f"        if transform is None:")
        lines.append(f"            from {mod} import {attr}")
        lines.append(f"            transform = {attr}")

    # --- api path ---
    api_path = path.lstrip("/")
    if id_param:
        api_path = api_path.replace("{" + id_param + "}", "{self.id}")
    lines.append(f"        _path = f\"{api_path}\"")

    # --- body dict ---
    if http_method == "POST" and mtype != "upload_parquet":
        if body_params:
            lines.append("        _body = {k: v for k, v in {")
            for bp in body_params:
                lines.append(f"            {bp['name']!r}: {bp['py_name']},")
            lines.append("        }.items() if v is not None}")
        else:
            lines.append("        _body = {}")

    # --- call ---
    if mtype == "upload_parquet":
        lines.append("        return self._send_parquet(_path, filepath)")

    elif mtype == "parquet_response":
        lines.append("        return self._load_parquet(_path, _body, transform=transform)")

    elif is_create:
        lines.append("        _inst = cls(org_id=org_id, api_key=api_key, base_url=base_url)")
        lines.append("        _resp = _inst._post(_path, _body)")
        if id_field:
            lines.append(f"        _inst.id = _resp.get({id_field!r})")
        lines.append("        return _inst")

    elif http_method == "GET":
        if response_key:
            lines.append("        _resp = self._get(_path)")
            nav = "_resp"
            for k in response_key:
                nav = f"{nav}[{k!r}]"
            lines.append("        import pandas as pd")
            lines.append(f"        _df = pd.DataFrame({nav})")
            lines.append("        return transform(_df) if transform else _df")
        else:
            lines.append("        return self._get(_path)")

    elif http_method == "POST":
        if response_key:
            lines.append("        _resp = self._post(_path, _body)")
            nav = "_resp"
            for k in response_key:
                nav = f"{nav}[{k!r}]"
            lines.append("        import pandas as pd")
            lines.append(f"        _df = pd.DataFrame({nav})")
            lines.append("        return transform(_df) if transform else _df")
        else:
            lines.append("        return self._post(_path, _body)")

    lines.append("")
    return "\n".join(lines)


def render_wait(entity_cfg: dict) -> str:
    """Emit a wait() method that polls the entity status until ready or failed."""
    wait_poll = entity_cfg.get("wait_poll")
    status_field = entity_cfg.get("wait_status_field", "status")
    ready = entity_cfg.get("wait_ready", "ready")
    failed = entity_cfg.get("wait_failed", ["failed"])
    id_param = entity_cfg.get("id_param", "")

    if not wait_poll:
        return ""

    _, poll_path = wait_poll.split(" ", 1)
    poll_path = poll_path.lstrip("/")
    if id_param:
        poll_path = poll_path.replace("{" + id_param + "}", "{self.id}")

    failed_set = repr(set(failed))

    lines = [
        "    def wait(self, poll_interval_start: float = 2.0, poll_interval_max: float = 60.0):",
        '        """Block until entity reaches ready or failed state. Returns self.',
        f"        → GET {poll_path}",
        '        """',
        "        import time",
        "        from .base import SturdyStatsSdkError",
        "        interval = poll_interval_start",
        "        while True:",
        f"            _resp = self._get(f\"{poll_path}\")",
        f"            _s = _resp.get({status_field!r})",
        f"            if _s == {ready!r}:",
        "                return self",
        f"            if _s in {failed_set}:",
        f"                raise SturdyStatsSdkError(0, f\"{{self}} failed with status '{{_s}}'\")",
        "            time.sleep(interval)",
        "            interval = min(interval * 1.5, poll_interval_max)",
        "",
    ]
    return "\n".join(lines)


def render_class(entity_cfg: dict, spec: dict) -> str:
    class_name = entity_cfg["class"]
    methods = entity_cfg["methods"]
    tag = entity_cfg.get("tag", class_name.lower())
    tag_desc = get_tag_description(spec, tag)

    typedefs = []
    method_blocks = [render_method(m, spec, entity_cfg, typedefs) for m in methods]

    header = [
        "# generated by scripts/gen_sdk.py — do not edit by hand",
        "from __future__ import annotations",
        "from typing import Any, Literal, Optional, TYPE_CHECKING, TypedDict",
        "from .base import SturdyStatsBase",
        "",
        "if TYPE_CHECKING:",
        "    import pandas as pd",
        "",
    ]
    if typedefs:
        header.append("")
        header.append("\n\n".join(typedefs))
    header += [
        "",
        "",
        f"class {class_name}(SturdyStatsBase):",
        f'    """{tag_desc}"""' if tag_desc else '    """See method docstrings for routes and parameter details."""',
        "",
    ]

    lines = header + method_blocks

    wait_method = render_wait(entity_cfg)
    if wait_method:
        lines.append(wait_method)

    return "\n".join(lines)


def main():
    config = json.loads(CONFIG_PATH.read_text())
    spec = load_openapi()
    OUT_DIR.mkdir(exist_ok=True)

    for entity in config["entities"]:
        class_name = entity["class"]
        out_path = OUT_DIR / f"{class_name.lower()}.py"
        print(f"  generating {out_path.name}...")
        out_path.write_text(render_class(entity, spec))
        print(f"  wrote {out_path}")

    print("done.")


if __name__ == "__main__":
    main()
