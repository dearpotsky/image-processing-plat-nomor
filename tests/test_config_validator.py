from app.pipeline.config import (
    PipelineConfig,
)

from app.pipeline.config_validator import (
    ConfigValidator,
    ConfigValidationError,
)


def test_valid_config():

    config = PipelineConfig()

    validator = ConfigValidator()

    result = validator.validate(
        config
    )

    assert result is True

    print(
        "✓ Valid configuration "
        "diterima."
    )


def test_invalid_confidence():

    config = PipelineConfig()

    config.detector.confidence = 1.5

    validator = ConfigValidator()

    try:

        validator.validate(config)

    except ConfigValidationError as error:

        print(
            "✓ Invalid confidence "
            "berhasil ditolak."
        )

        print(error)

    else:

        raise AssertionError(
            "Confidence 1.5 seharusnya "
            "ditolak."
        )


def test_invalid_target_size():

    config = PipelineConfig()

    config.preprocessing.target_width = 0

    validator = ConfigValidator()

    try:

        validator.validate(config)

    except ConfigValidationError:

        print(
            "✓ Invalid target width "
            "berhasil ditolak."
        )

    else:

        raise AssertionError(
            "Target width 0 seharusnya "
            "ditolak."
        )


def test_invalid_enhancement():

    config = PipelineConfig()

    config.enhancement.method = (
        "metode_ngawur"
    )

    validator = ConfigValidator()

    try:

        validator.validate(config)

    except ConfigValidationError:

        print(
            "✓ Invalid enhancement "
            "berhasil ditolak."
        )

    else:

        raise AssertionError(
            "Enhancement method "
            "seharusnya ditolak."
        )


def test_invalid_segmentation_method():

    config = PipelineConfig()

    config.segmentation.method = (
        "metode_ngawur"
    )

    validator = ConfigValidator()

    try:

        validator.validate(config)

    except ConfigValidationError:

        print(
            "✓ Invalid segmentation "
            "berhasil ditolak."
        )

    else:

        raise AssertionError(
            "Segmentation method "
            "seharusnya ditolak."
        )


def test_invalid_adaptive_block_size():

    config = PipelineConfig()

    config.segmentation.method = (
        "adaptive"
    )

    config.segmentation.block_size = 10

    validator = ConfigValidator()

    try:

        validator.validate(config)

    except ConfigValidationError:

        print(
            "✓ Adaptive block_size "
            "genap berhasil ditolak."
        )

    else:

        raise AssertionError(
            "Adaptive block_size 10 "
            "seharusnya ditolak."
        )


def test_invalid_connectivity():

    config = PipelineConfig()

    config.feature.connectivity = 6

    validator = ConfigValidator()

    try:

        validator.validate(config)

    except ConfigValidationError:

        print(
            "✓ Invalid connectivity "
            "berhasil ditolak."
        )

    else:

        raise AssertionError(
            "Connectivity 6 "
            "seharusnya ditolak."
        )


def test_invalid_min_area():

    config = PipelineConfig()

    config.feature.min_area = -10

    validator = ConfigValidator()

    try:

        validator.validate(config)

    except ConfigValidationError:

        print(
            "✓ Negative min_area "
            "berhasil ditolak."
        )

    else:

        raise AssertionError(
            "min_area negatif "
            "seharusnya ditolak."
        )


def main():

    print("=" * 60)
    print("CONFIG VALIDATOR TEST")
    print("=" * 60)

    test_valid_config()

    test_invalid_confidence()

    test_invalid_target_size()

    test_invalid_enhancement()

    test_invalid_segmentation_method()

    test_invalid_adaptive_block_size()

    test_invalid_connectivity()

    test_invalid_min_area()

    print("\n" + "=" * 60)
    print(
        "SEMUA CONFIG VALIDATION TEST BERHASIL"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()