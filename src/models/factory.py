from .resnet_attr import ResNetAttrModel
from .efficientnet_attr import EfficientNetAttrModel
from .vit_attr import ViTAttrModel
from .swin_attr import SwinAttrModel
from .pcb_attr import PCBAttrModel


def build_model(model_name, num_classes, pretrained=True, **kwargs):
    model_name = model_name.lower()

    if model_name == "resnet50":
        return ResNetAttrModel(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=kwargs.get("dropout", 0.5)
        )

    if model_name == "efficientnet_b0":
        return EfficientNetAttrModel(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=kwargs.get("dropout", 0.3)
        )

    if model_name == "vit_b16":
        return ViTAttrModel(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=kwargs.get("dropout", 0.1)
        )

    if model_name == "swin_t":
        return SwinAttrModel(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=kwargs.get("dropout", 0.1)
        )

    if model_name == "pcb":
        return PCBAttrModel(
            num_classes=num_classes,
            pretrained=pretrained,
            num_parts=kwargs.get("num_parts", 6),
            dropout=kwargs.get("dropout", 0.5),
            reduced_dim=kwargs.get("reduced_dim", 256)
        )

    raise ValueError(f"Unsupported model: {model_name}")