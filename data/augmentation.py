import cv2
import torchvision.transforms as tfs


def Common(image):
    image = cv2.equalizeHist(image)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    return image


def Aug(image):
    transform = tfs.Compose([
        tfs.RandomAffine(degrees=(-15, 15), translate=(0.05, 0.05),
                         scale=(0.95, 1.05), fillcolor=128)
    ])
    return transform(image)


def get_transforms(image, target=None, ttype='common'):
    assert target is None, "unsupported argument 'target'"
    ttype = ttype.strip()
    if ttype == 'Common':
        return Common(image)
    elif ttype == 'None':
        return image
    elif ttype == 'Aug':
        return Aug(image)
    else:
        raise ValueError(f"Unknown ttype '{ttype}")
