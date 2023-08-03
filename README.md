# Epigos Python

![Tests](https://github.com/Epigos-Inc/epigos-python/actions/workflows/tests.yaml/badge.svg)

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

## Gettting Started

To make your first API call, you will need to signup at [epigos.ai](https://epigos.ai) and create an
API key for your workspace. Please contact our sales team for a demo.

### Initialization:

```python
import epigos

client = epigos.Epigos("api_key")
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
print(results.json())
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
