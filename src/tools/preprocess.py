import os
from PIL import Image


file_path = os.path.dirname(__file__)
base_path = os.path.dirname(file_path)

# Original dataset
data_path = os.path.join(base_path, "data")

rgb_path = os.path.join(data_path, "visible")
thermal_path = os.path.join(data_path, "infrared")

rgb_train = os.path.join(rgb_path, "train")
rgb_test = os.path.join(rgb_path, "test")

thermal_train = os.path.join(thermal_path, "train")
thermal_test = os.path.join(thermal_path, "test")


# Processed dataset
processed_data_path = os.path.join(base_path, "processed_data")

processed_rgb_path = os.path.join(processed_data_path, "visible")
processed_thermal_path = os.path.join(processed_data_path, "infrared")

processed_rgb_train = os.path.join(processed_rgb_path, "train")
processed_rgb_test = os.path.join(processed_rgb_path, "test")

processed_thermal_train = os.path.join(processed_thermal_path, "train")
processed_thermal_test = os.path.join(processed_thermal_path, "test")


def resize(image):
    return image.resize(
        size=(256, 256),
        resample=Image.LANCZOS
    )


def process_folder(input_path, output_path):

    os.makedirs(output_path, exist_ok=True)

    image_list = sorted(os.listdir(input_path))

    for image in image_list:

        image_path = os.path.join(input_path, image)

        img = Image.open(image_path)
        img = resize(img)

        save_path = os.path.join(output_path, image)
        img.save(save_path)

        img.close()


def main():

    process_folder(rgb_train, processed_rgb_train)
    process_folder(rgb_test, processed_rgb_test)

    process_folder(thermal_train, processed_thermal_train)
    process_folder(thermal_test, processed_thermal_test)


if __name__ == "__main__":
    main()