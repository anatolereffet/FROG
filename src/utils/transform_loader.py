import albumentations as A
from albumentations.pytorch import ToTensorV2


init_transforms = [
    A.ToGray(p=1),
    A.Resize(height=96, width=96),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
]
horizontal_transform = [
    A.ToGray(p=1),
    A.Resize(height=96, width=96),
    A.HorizontalFlip(always_apply=True),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
]
rotation_transform = [
    A.ToGray(p=1),
    A.Resize(height=96, width=96),
    A.Rotate(limit=(-15, 15), always_apply=True),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
]


basic_transform = A.Compose(init_transforms)
horizontal_transform = A.Compose(horizontal_transform)
rotation_transform = A.Compose(rotation_transform)
