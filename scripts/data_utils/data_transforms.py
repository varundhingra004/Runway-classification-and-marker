from PIL import Image
from torchvision import transforms

# ===================================================================
# DATA PROCESSING AND DATA AUGMENTATION PROCESS FOR TRAINING DATA
# ===================================================================


# LEARNINGS: The Compose method  of torchvision.transforms stitches the different transformations together as a sequence.
# Thus, the output of the first transform becomes output to the second transform and so on.

train_transforms = transforms.Compose([

    # Resize FIRST (keep consistent input)
    transforms.Resize((224, 224)),  # larger size for better Grad-CAM resolution

    # --- GEOMETRIC (SAFE) ---
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),

    # --- LIGHT POSITIONAL VARIATION ---
    transforms.RandomApply([
        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),   # reduced
            scale=(0.95, 1.05),       # tighter
            shear=5                   # reduced
        )
    ], p=0.3),

    # --- COLOR ---
    transforms.RandomApply([
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05
        )
    ], p=0.3),

    # --- PREPROCESSING ---
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

validate_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225]
    )
])

def apply_transforms(image_paths, transform):
    imgs = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        imgs.append(transform(img))
    return imgs
# =========================================================
# 🔬 OPTIONAL (FOR LATER EXPERIMENTATION)
# =========================================================

"""
You can try RandomResizedCrop later AFTER verifying Grad-CAM results.

IMPORTANT:
RandomResizedCrop randomly selects a region of the image and resizes it.
The 'scale' parameter controls how much of the original image is kept.
(Default is (0.08, 1.0) — which can crop very small regions) :contentReference[oaicite:0]{index=0}

Safe version for your project:

transforms.RandomResizedCrop(
    size=224,
    scale=(0.85, 1.0)   # keeps 85%–100% of image → runway likely preserved
)

WARNING:
If scale is too small (e.g., 0.5 or lower),
the runway may be completely removed from the image,
which can hurt both:
- model learning
- Grad-CAM heatmaps

Recommended workflow:
1. Train WITHOUT cropping (current setup)
2. Generate Grad-CAM → verify runway focus
3. THEN introduce controlled cropping if needed
"""