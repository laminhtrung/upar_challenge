from .resnet_attr import ResNetAttrModel
from .efficientnet_attr import EfficientNetAttrModel
from .vit_attr import ViTAttrModel
from .swin_attr import SwinAttrModel
from .pcb_attr import PCBAttrModel
from .pcb_real import PCBRealAttrModel
from .hybrid_effb0_transformer_v3 import HybridEfficientNetTransformerV3
from .resnet_update import ResNetUpdateModel
from .efficientnet_update import EfficientNetUpdateModel

from .hybrid_effb0_transformer import HybridEfficientNetTransformer
from .dual_branch_effb0_swin import DualBranchEfficientNetSwin


def build_model(model_name, num_classes, pretrained=True, **kwargs):
    model_name = model_name.lower()

    if model_name == "resnet50":
        return ResNetAttrModel(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=kwargs.get("dropout", 0.5)
        )
        
    elif model_name == "resnet_update":
        return ResNetUpdateModel(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=kwargs.get("dropout", 0.5)
        )

    elif model_name == "efficientnet_b0":
        return EfficientNetAttrModel(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=kwargs.get("dropout", 0.3)
        )

    elif model_name == "efficientnet_update":
        return EfficientNetUpdateModel(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=kwargs.get("dropout", 0.3)
        )

    elif model_name == "vit_b16":
        return ViTAttrModel(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=kwargs.get("dropout", 0.1)
        )

    elif model_name == "swin_t":
        return SwinAttrModel(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=kwargs.get("dropout", 0.1)
        )

    elif model_name == "pcb":
        return PCBAttrModel(
            num_classes=num_classes,
            pretrained=pretrained,
            num_parts=kwargs.get("num_parts", 6),
            dropout=kwargs.get("dropout", 0.5),
            reduced_dim=kwargs.get("reduced_dim", 256)
        )

    elif model_name == "hybrid_effb0_transformer":
        return HybridEfficientNetTransformer(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=kwargs.get("dropout", 0.4),
            backbone_name=kwargs.get("backbone_name", "efficientnet_b0"),
            d_model=kwargs.get("d_model", 256),
            nhead=kwargs.get("nhead", 8),
            num_transformer_layers=kwargs.get("num_transformer_layers", 2),
            dim_feedforward=kwargs.get("dim_feedforward", 512),
            transformer_dropout=kwargs.get("transformer_dropout", 0.1),
        )

    elif model_name == "dual_branch_effb0_swin":
        return DualBranchEfficientNetSwin(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=kwargs.get("dropout", 0.4),
            cnn_name=kwargs.get("cnn_name", "efficientnet_b0"),
            transformer_name=kwargs.get("transformer_name", "swin_tiny_patch4_window7_224"),
            fusion_dim=kwargs.get("fusion_dim", 256),
            image_size=kwargs.get("image_size", (384, 128)),
        )
        
    elif model_name == "pcb_real":
        return PCBRealAttrModel(
            num_classes=num_classes,
            pretrained=pretrained,
            backbone_name=kwargs.get("backbone_name", "resnet50"),
            num_parts=kwargs.get("num_parts", 6),
            reduced_dim=kwargs.get("reduced_dim", 256),
            dropout=kwargs.get("dropout", 0.5)
        )
    elif model_name == "hybrid_effb0_transformer_v3":
        return HybridEfficientNetTransformerV3(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=kwargs.get("dropout", 0.4),
            backbone_name=kwargs.get("backbone_name", "efficientnet_b0"),
            d_model=kwargs.get("d_model", 256),
            nhead=kwargs.get("nhead", 8),
            num_transformer_layers=kwargs.get("num_transformer_layers", 2),
            dim_feedforward=kwargs.get("dim_feedforward", 512),
            transformer_dropout=kwargs.get("transformer_dropout", 0.1),
            num_parts=kwargs.get("num_parts", 3),
        )
    
    elif model_name == "effb0_part":
        return EfficientNetB0PartAttrModel(
            num_classes=num_classes,
            pretrained=pretrained,
            num_parts=kwargs.get("num_parts", 4),
            embed_dim=kwargs.get("embed_dim", 256),
            dropout=kwargs.get("dropout", 0.2)
        )


    raise ValueError(f"Unsupported model: {model_name}")