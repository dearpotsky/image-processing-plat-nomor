from dataclasses import asdict
import json


def config_to_dict(config):
    return asdict(config)


def save_config_json(
    config,
    output_path,
):
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = config_to_dict(config)

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False,
        )