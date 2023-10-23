import typing
import xml.etree.ElementTree as ET


def read_pascal_voc_to_coco(
    annotation_file: str, img_scale: typing.Optional[typing.Tuple[int, int]] = None
) -> typing.Dict[str, typing.Any]:
    """
    Reads Pascal VOC [xmin, ymin, xmax, ymax] annotations from file and converts it to COCO-style annotations.
    :param annotation_file: Path to annotation file to read.
    :param img_scale: Image size to rescale annotations.
    :return: Dict: Contains image metadata and COCO bounding boxes (x, y, width, height).
    """
    tree = ET.parse(annotation_file)
    root = tree.getroot()

    image_width = int(root.find("size")[0].text)
    image_height = int(root.find("size")[1].text)

    annotation = {
        "filename": root.find("filename").text,
        "metadata": {"image": {"width": image_width, "height": image_height}},
        "boxes": [],
        "labels": set(),
    }

    for obj in root.iter("object"):
        name = obj.find("name").text
        bb = obj.find("bndbox")

        xmin = int(float(bb.find("xmin").text))
        ymin = int(float(bb.find("ymin").text))
        xmax = int(float(bb.find("xmax").text))
        ymax = int(float(bb.find("ymax").text))

        width = xmax - xmin
        height = ymax - ymin

        if img_scale:
            # scale annotations to fit image upload size
            scale_x = img_scale[0] / image_width
            scale_y = img_scale[1] / image_height
            xmin *= scale_x
            ymin *= scale_y
            width *= scale_x
            height *= scale_y

        box = dict(
            label=name, x=int(xmin), y=int(ymin), width=int(width), height=int(height)
        )
        annotation["boxes"].append(box)
        annotation["labels"].add(name)

    annotation["labels"] = sorted(list(annotation["labels"]))
    return annotation
