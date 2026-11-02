
import re
import shutil
import tarfile
from pathlib import Path
from tqdm import tqdm


SOURCE = Path(r"E:\ILSVRC2012_img_train.tar")
OUTPUT = Path(r"D:\emoji_optimizer\imagenet\ILSVRC2012_img_train")


def extract_imagenet_stream():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    class_count = 0
    image_count = 0
    failed_classes = []

    print("=" * 65)
    print("ImageNet streaming extraction")
    print("=" * 65)
    print(f"Source:      {SOURCE}")
    print(f"Destination: {OUTPUT}")
    print("Intermediate TAR files: None")
    print("=" * 65)

    # Stream the outer ImageNet TAR archive.
    with tarfile.open(SOURCE, mode="r|*") as outer_tar:

        for outer_member in outer_tar:

            # Only process class archives such as n01440764.tar.
            match = re.fullmatch(
                r"(n\d{8})\.tar",
                Path(outer_member.name).name
            )

            if not match or not outer_member.isfile():
                continue

            class_name = match.group(1)
            class_dir = OUTPUT / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n[{class_count + 1}] Extracting {class_name} ...",
                  flush=True)

            try:
                # Read the nested TAR directly from the outer TAR.
                inner_file = outer_tar.extractfile(outer_member)

                if inner_file is None:
                    raise RuntimeError("Cannot read nested TAR archive")

                class_images = 0

                with inner_file:
                    with tarfile.open(
                        fileobj=inner_file,
                        mode="r|*"
                    ) as inner_tar:

                        for member in inner_tar:

                            # Extract regular image files only.
                            if not member.isfile():
                                continue

                            # Prevent directory traversal.
                            filename = Path(member.name).name

                            if not filename:
                                continue

                            if not filename.lower().endswith(
                                (".jpeg", ".jpg")
                            ):
                                continue

                            source_file = inner_tar.extractfile(member)

                            if source_file is None:
                                continue

                            destination_file = class_dir / filename

                            with source_file:
                                with open(destination_file, "wb") as output:
                                    shutil.copyfileobj(
                                        source_file,
                                        output,
                                        length=1024 * 1024
                                    )

                            class_images += 1

                class_count += 1
                image_count += class_images

                print(
                    f"Completed {class_name}: "
                    f"{class_images:,} images | "
                    f"Total: {image_count:,}",
                    flush=True
                )

            except Exception as error:
                failed_classes.append(class_name)

                print(
                    f"ERROR in {class_name}: {error}",
                    flush=True
                )

                raise

    print("\n" + "=" * 65)
    print("EXTRACTION FINISHED")
    print("=" * 65)
    print(f"Classes extracted: {class_count}")
    print(f"Images extracted:  {image_count:,}")
    print(f"Failed classes:    {len(failed_classes)}")

    if failed_classes:
        print("Failed:", failed_classes)

    print(f"Output directory: {OUTPUT}")


if __name__ == "__main__":
    extract_imagenet_stream()
