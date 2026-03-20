from torchvision import transforms


def build_transforms(image_size=(384, 128), is_train=True):
    h, w = image_size

    if is_train:
        return transforms.Compose([
            transforms.Resize((h, w)),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(p=0.4),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    return transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])