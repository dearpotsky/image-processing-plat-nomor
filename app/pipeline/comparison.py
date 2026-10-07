from dataclasses import dataclass


@dataclass
class ExperimentComparison:
    name: str
    image_count: int
    plate_count: int

    avg_foreground_ratio: float
    avg_mean_intensity: float
    avg_std_intensity: float

    avg_total_components: float
    avg_valid_components: float


class ExperimentComparator:

    def compare(
        self,
        experiment_results,
    ):

        comparisons = []

        for experiment in (
            experiment_results
        ):

            batch_result = (
                experiment.result
            )

            foreground_values = []
            mean_values = []
            std_values = []

            total_component_values = []
            valid_component_values = []

            for image_result in (
                batch_result.images
            ):

                for plate in (
                    image_result.plates
                ):

                    features = (
                        plate.features
                    )

                    if (
                        "foreground_ratio"
                        in features
                    ):
                        foreground_values.append(
                            features[
                                "foreground_ratio"
                            ]
                        )

                    if (
                        "mean_intensity"
                        in features
                    ):
                        mean_values.append(
                            features[
                                "mean_intensity"
                            ]
                        )

                    if (
                        "std_intensity"
                        in features
                    ):
                        std_values.append(
                            features[
                                "std_intensity"
                            ]
                        )

                    if (
                        "total_components"
                        in features
                    ):
                        total_component_values.append(
                            features[
                                "total_components"
                            ]
                        )

                    if (
                        "valid_components"
                        in features
                    ):
                        valid_component_values.append(
                            features[
                                "valid_components"
                            ]
                        )

            comparisons.append(
                ExperimentComparison(

                    name=experiment.name,

                    image_count=(
                        batch_result.image_count
                    ),

                    plate_count=(
                        batch_result
                        .total_plate_count
                    ),

                    avg_foreground_ratio=(
                        self._average(
                            foreground_values
                        )
                    ),

                    avg_mean_intensity=(
                        self._average(
                            mean_values
                        )
                    ),

                    avg_std_intensity=(
                        self._average(
                            std_values
                        )
                    ),

                    avg_total_components=(
                        self._average(
                            total_component_values
                        )
                    ),

                    avg_valid_components=(
                        self._average(
                            valid_component_values
                        )
                    ),
                )
            )

        return comparisons

    @staticmethod
    def _average(values):

        if not values:
            return 0.0

        return sum(values) / len(values)