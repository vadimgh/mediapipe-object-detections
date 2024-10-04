import os

from mediapipe_model_maker import object_detector


def train_model():
    train_dataset_path = os.path.join("train")
    validation_dataset_path = os.path.join("validation")
    train_data = object_detector.Dataset.from_coco_folder(
        train_dataset_path, cache_dir="/tmp/od_data/train"
    )
    validation_data = object_detector.Dataset.from_coco_folder(
        validation_dataset_path, cache_dir="/tmp/od_data/validation"
    )
    print("train_data size: ", train_data.size)
    print("validation_data size: ", validation_data.size)

    model_export_path = validation_dataset_path = os.path.join("model")

    spec = object_detector.SupportedModels.MOBILENET_MULTI_AVG
    hparams = object_detector.HParams(export_dir=model_export_path)
    options = object_detector.ObjectDetectorOptions(
        supported_model=spec, hparams=hparams
    )

    # train
    model = object_detector.ObjectDetector.create(
        train_data=train_data, validation_data=validation_data, options=options
    )

    loss, coco_metrics = model.evaluate(validation_data, batch_size=4)
    print(f"Validation loss: {loss}")
    print(f"Validation coco metrics: {coco_metrics}")

    model.export_model()


train_model()
