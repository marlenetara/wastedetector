from pathlib import Path
import random
import shutil


# ============================================================
# PATHS
# ============================================================

SOURCE_DIR = Path(
    r"C:\Users\Marlene Tara\WasteDetector\WasteDetector\dataset"
)

OUTPUT_DIR = Path(
    r"C:\Users\Marlene Tara\WasteDetector\WasteDetector\dataset_16class"
)


# ============================================================
# SETTINGS
# ============================================================

TRAIN_RATIO = 0.8

IMAGE_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
]

random.seed(42)


# ============================================================
# FIND ALL SUBCATEGORIES
# ============================================================

categories = []

for parent_folder in SOURCE_DIR.iterdir():

    if not parent_folder.is_dir():
        continue

    for category_folder in parent_folder.iterdir():

        if category_folder.is_dir():
            categories.append(category_folder)


print("Found categories:")

for category in categories:
    print(f"  {category.name}")


# ============================================================
# CREATE TRAIN / VAL FOLDERS
# ============================================================

train_dir = OUTPUT_DIR / "train"
val_dir = OUTPUT_DIR / "val"

train_dir.mkdir(
    parents=True,
    exist_ok=True
)

val_dir.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# COPY IMAGES
# ============================================================

for category_folder in categories:

    category_name = category_folder.name

    train_category = (
        train_dir / category_name
    )

    val_category = (
        val_dir / category_name
    )

    train_category.mkdir(
        parents=True,
        exist_ok=True
    )

    val_category.mkdir(
        parents=True,
        exist_ok=True
    )

    # Find images
    images = []

    for file in category_folder.rglob("*"):

        if (
            file.is_file()
            and file.suffix.lower()
            in IMAGE_EXTENSIONS
        ):

            images.append(file)

    # Shuffle images
    random.shuffle(images)

    # Calculate split
    split_index = int(
        len(images) * TRAIN_RATIO
    )

    train_images = images[:split_index]
    val_images = images[split_index:]

    print(
        f"\n{category_name}: "
        f"{len(images)} images"
    )

    print(
        f"  Train: {len(train_images)}"
    )

    print(
        f"  Validation: {len(val_images)}"
    )

    # --------------------------------------------------------
    # Copy training images
    # --------------------------------------------------------

    for number, image in enumerate(train_images):

        destination = (
            train_category /
            f"{number}_{image.name}"
        )

        shutil.copy2(
            image,
            destination
        )

    # --------------------------------------------------------
    # Copy validation images
    # --------------------------------------------------------

    for number, image in enumerate(val_images):

        destination = (
            val_category /
            f"{number}_{image.name}"
        )

        shutil.copy2(
            image,
            destination
        )


# ============================================================
# FINISHED
# ============================================================

print("\n========================================")
print("DATASET PREPARATION COMPLETE")
print("========================================")

print(
    f"\nCreated dataset at:\n{OUTPUT_DIR}"
)

print(
    f"\nNumber of classes: {len(categories)}"
)

print("\nClasses:")

for category in categories:
    print(
        f"  - {category.name}"
    )

print(
    "\nYou can now train the model."
)
