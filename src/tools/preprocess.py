import os
from PIL import Image
import time

file_path = os.path.dirname(__file__) #src/tools/preprocess.py
base_path = os.path.dirname(os.path.dirname(file_path)) # root 

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

    #print("INPUT:", input_path)
    #print("ABSOLUTE:", os.path.abspath(input_path))
    #print("EXISTS:", os.path.exists(input_path))

    if not os.path.exists(input_path):
        print("preprocess - input path problem")
        return
    
    os.makedirs(output_path, exist_ok=True)

    image_list = sorted(os.listdir(input_path))
    total = len(image_list)
    start_time = time.time()

    for i, image in enumerate(image_list, 1):

        image_path = os.path.join(input_path, image)

        if not os.path.isfile(image_path):
            continue

        try:
            img = Image.open(image_path)
            img = resize(img)

            save_path = os.path.join(output_path, image)
            img.save(save_path)

            img.close()

        except Exception as e: 
            print(f"error: {e}")

        if i % 100 == 0 or i == total:
            elapsed = time.time() - start_time
            rate = i / elapsed
            remaining = total - i
            eta = remaining / rate if rate > 0 else 0

            print(
                f"\rProcessed: {i}/{total} "
                f"({i / total * 100:.1f}%) | "
                f"Elapsed: {elapsed:.1f}s | "
                f"Rate: {rate:.1f} img/s | "
                f"ETA: {eta:.1f}s",
                end=""
            )

def main():

    process_folder(rgb_train, processed_rgb_train)
    process_folder(rgb_test, processed_rgb_test)

    process_folder(thermal_train, processed_thermal_train)
    process_folder(thermal_test, processed_thermal_test)


if __name__ == "__main__":
    main()