from typing import Any


def remove_field_recursively(d: Any, field: str) -> Any:
    if isinstance(d, dict):
        if field in d:
            del d[field]
        for k in d.keys():
            d[k] = remove_field_recursively(d[k], field)
        return d
    elif isinstance(d, list):
        return [remove_field_recursively(e, field) for e in d]
    else:
        return d


def flatten_single_element_allof(d: Any) -> Any:
    if isinstance(d, dict):
        if "allOf" in d and len(d["allOf"]) == 1:
            for k, v in d["allOf"][0].items():
                if k not in d:
                    d[k] = v
            del d["allOf"]
        for k in d.keys():
            d[k] = flatten_single_element_allof(d[k])
        return d
    elif isinstance(d, list):
        return [flatten_single_element_allof(e) for e in d]
    else:
        return d


def delete_null_field(d: Any) -> Any:
    if isinstance(d, dict):
        remove_keys = []
        for k in d.keys():
            if d[k] is None:
                remove_keys.append(k)
            else:
                d[k] = delete_null_field(d[k])
        for k in remove_keys:
            del d[k]
        return d
    elif isinstance(d, list):
        return [delete_null_field(e) for e in d]
    else:
        return d
