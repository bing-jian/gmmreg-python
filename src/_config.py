#!/usr/bin/env python
#coding=utf-8

"""Config file loading shared by run_config() and the demo/plotting scripts.

Both the classic .ini configs (parsed with the stdlib configparser) and
.yaml/.yml configs (parsed with PyYAML) are supported. Both formats mirror
the same section layout: a 'FILES' section listing input/output point-set
paths and a 'GMMREG_OPT' section with optimization parameters.
"""

import os
import configparser
import yaml


def _is_yaml(path):
    return os.path.splitext(path)[1].lower() in ('.yaml', '.yml')


def read_sections(f_config):
    """Read a config file and return its sections as a dict of dicts.

    For .ini files, values come back as strings (as configparser returns
    them). For .yaml/.yml files, values keep their native YAML types
    (int, float, list, ...).
    """
    if _is_yaml(f_config):
        with open(f_config) as fh:
            data = yaml.safe_load(fh) or {}
        return {section: dict(values) for section, values in data.items()}
    c = configparser.ConfigParser()
    c.read(f_config)
    return {section: dict(c.items(section)) for section in c.sections()}


def as_list(value, caster):
    """Coerce a config value into a list of `caster`-typed elements.

    Accepts either a native YAML list or a whitespace-separated string (the
    representation used by .ini files, e.g. sigma = .3 .2 .1 .05).
    """
    if isinstance(value, str):
        return [caster(v) for v in value.split()]
    return [caster(v) for v in value]
