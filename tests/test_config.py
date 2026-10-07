from app.pipeline.config import PipelineConfig


def main():
    config = PipelineConfig()

    print("=" * 60)
    print("PIPELINE CONFIG TEST")
    print("=" * 60)

    print("\nPreprocessing")
    print("Target size :", config.preprocessing.target_size)
    print("Grayscale   :", config.preprocessing.grayscale)
    print("Normalize   :", config.preprocessing.normalize)

    print("\nEnhancement")
    print("Method      :", config.enhancement.method)
    print("Clip limit  :", config.enhancement.clip_limit)
    print("Tile grid   :", config.enhancement.tile_grid_size)

    print("\nSegmentation")
    print("Method      :", config.segmentation.method)
    print("Block size  :", config.segmentation.block_size)
    print("C           :", config.segmentation.c)

    print("\nFeature")
    print(
        "Connected components :",
        config.feature.connected_components,
    )
    print("Min area              :", config.feature.min_area)


if __name__ == "__main__":
    main()