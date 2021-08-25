import yaml
from addict import Dict
import pydoc


def object_from_dict(d, parent=None, **default_kwargs):
    kwargs = d.copy()
    object_type = kwargs.pop("type")

    for name, value in default_kwargs.items():
        kwargs.setdefault(name, value)

    if parent is not None:
        return getattr(parent, object_type)(**kwargs)
    else:
        return pydoc.locate(object_type)(**kwargs)


def get_param_from_config(train_classifier_model_config_path: str) -> dict:
    with open(train_classifier_model_config_path, 'r') as config:
        data_config = yaml.safe_load(config)

    data_config = Dict(data_config)

    return data_config
