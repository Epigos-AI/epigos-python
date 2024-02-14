# Epigos Python

![Tests](https://github.com/Epigos-AI/epigos-python/actions/workflows/tests.yaml/badge.svg)

Epigos provides an end-to-end platform to annotate data, train computer vision AI models,
deploy them seamlessly and host the models via API's.

For more details, visit [epigos.ai](https://epigos.ai).

The Epigos Python Package is a python wrapper around the core Epigos AI web application and REST API.

## Installation

To install this package, please use `Python 3.9` or higher.

To add `epigos` to your project simply install with pip:

```shell
pip install epigos
```

Or with poetry

```shell
poetry add epigos
```

## Getting Started

To make your first API call, you will need to signup at [epigos.ai](https://epigos.ai) and create an
API key for your workspace. Please contact our sales team for a demo.

### Initialization:

```python
import epigos

client = epigos.Epigos("api_key")
```

### Project:

Manage project and upload dataset into your project using the  `Project ID`.

#### Upload an image with annotation

```python
import epigos

client = epigos.Epigos("api_key")

# load project
project = client.project("project_id")

# upload image with Pascal VOC annotation
record = project.upload("path/to/image.jpg", annotation_path="path/to/image.xml", box_format="pascal_voc")
print(record)

# upload image with YOLO annotation
record = project.upload("path/to/image.jpg", annotation_path="path/to/image.txt", box_format="yolo")
print(record)
```

#### Upload an entire dataset folder

```python
import epigos

client = epigos.Epigos("api_key")

# load project
project = client.project("project_id")

# upload Pascal VOC annotation dataset
records = project.upload_dataset("path/to/folder", box_format="pascal_voc")
print(tuple(records))

# upload YOLO annotation dataset
records = project.upload_dataset("path/to/folder", box_format="yolo")
print(tuple(records))
```

### Prediction:

Make predictions with any of the models deployed in your workspace using the `Model ID`.

#### Classification

```python
import epigos

client = epigos.Epigos("api_key")

# load classification model
model = client.classification("model_id")

# make predictions
results = model.predict("path/to/your/image.jpg")
print(results.dict())
```

#### Object detection

```python
import epigos

client = epigos.Epigos("api_key")

# load object detection model
model = client.object_detection("model_id")

# make predictions
results = model.detect("path/to/your/image.jpg")
print(results.dict())
# visualize detections
results.show()
```

## Contributing

If you want to extend our Python library or if you find a bug, please open a PR!

Also be sure to test your code with the `make` command at the root level directory.

Run tests:

```bash
make test
```

### Commit message guidelines

It’s important to write sensible commit messages to help the team move faster.

Please follow the [commit guidelines](https://www.conventionalcommits.org/en/v1.0.0/)

### Versioning

This project uses [Semantic Versioning](https://semver.org/).

## Publishing

This project is published on PyPi

## License

This library is released under the [MIT License](LICENSE).
