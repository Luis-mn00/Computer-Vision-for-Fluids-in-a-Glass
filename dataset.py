import cv2
import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import os
import shutil
from torch.utils.data import Dataset, random_split

from utils import resize_images_in_folder

class videos:
    def __init__(self, img_size):
        self.videos_dir = "data/videos"
        self.frames_dir = "data/frames"
        self.inputs_dir = "data/inputs"
        self.masks_dir = "data/masks"
        self.labels_dir = "data/labels"
        
        if not os.path.exists(self.videos_dir):
            print(f"Error: Videos directory '{self.videos_dir}' does not exist.")
            self.num_videos = 0
        else:
            self.num_videos = len(os.listdir(self.videos_dir))
            
        if not os.path.exists(self.inputs_dir):
            print(f"Warning: Videos directory '{self.inputs_dir}' does not exist.")
            self.num_inputs = 0
        else:
            self.num_inputs = len(os.listdir(self.inputs_dir))
            
        if not os.path.exists(self.masks_dir):
            print(f"Warning: Videos directory '{self.masks_dir}' does not exist.")
            self.num_masks = 0
        else:
            self.num_masks = len(os.listdir(self.masks_dir))
            
        self.img_size = img_size  # (256, 256)

    def extract_frames_individual(self, video_path, index):
        # Create output folder if it doesn't exist
        frame_path = os.path.join(self.frames_dir, f"frames_{index}")
        if not os.path.exists(frame_path):
            os.makedirs(frame_path)

        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video file {video_path}")
            return

        frame_count = 0
        saved_frame_count = 0  # Track the number of successfully saved frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # Break the loop if no frame is read

            # Resize the frame
            try:
                resized_frame = cv2.resize(frame, self.img_size)

                # Convert the frame to grayscale
                grayscale_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

                # Save the frame as an image
                frame_filename = os.path.join(frame_path, f"frame_{index}_{frame_count:04d}.png")
                if cv2.imwrite(frame_filename, grayscale_frame):
                    saved_frame_count += 1
                else:
                    print(f"Warning: Failed to save frame {frame_count} to {frame_filename}")
            except Exception as e:
                print(f"Error processing frame {frame_count}: {e}")

            frame_count += 1

        cap.release()
        print(f"Extracted {frame_count} frames from video {video_path}. Successfully saved {saved_frame_count} frames to folder {frame_path}")

    def extract_frames(self):
        if not os.path.exists(self.videos_dir) or len(os.listdir(self.videos_dir)) == 0:
            print(f"Error: No videos found in directory '{self.videos_dir}'.")
            return

        index = 1
        for video_name in os.listdir(self.videos_dir):
            video_path = os.path.join(self.videos_dir, video_name)

            # Extract frames from the video
            self.extract_frames_individual(video_path, index)
            index += 1
            
    def create_input_folder(self, num_frames_per_video=30):
        """
        Create a new input folder containing a random selection of frames from each video.
        """
        if self.num_videos == 0:
            print("Error: No videos found.")
            return
        
        # Create the new input folder
        input_folder_index = self.num_inputs + 1
        input_folder_path = os.path.join(self.inputs_dir, f"input_{input_folder_index}")
        os.makedirs(input_folder_path, exist_ok=True)
        self.num_inputs += 1

        for video_index in range(1, self.num_videos + 1):
            # Path to the frames of the current video
            frame_folder = os.path.join(self.frames_dir, f"frames_{video_index}")
            if not os.path.exists(frame_folder):
                continue

            # Get all frame filenames
            frame_files = [f for f in os.listdir(frame_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if len(frame_files) == 0:
                continue

            # Randomly select frames
            selected_frames = np.random.choice(frame_files, min(num_frames_per_video, len(frame_files)), replace=False)

            # Copy selected frames to the new input folder
            for frame_file in selected_frames:
                src_path = os.path.join(frame_folder, frame_file)
                dst_path = os.path.join(input_folder_path, frame_file)
                shutil.copy(src_path, dst_path)  # Copy the file to the input folder

        print(f"Created input folder: {input_folder_path}")
                
            
    def create_masks_from_input(self, index, surface_threshold=3, fluid_threshold=253):
        input_folder = os.path.join(self.inputs_dir, f"input_{index}")
        output_folder = os.path.join(self.masks_dir, f"masks_{index}")
        labels_folder = self.labels_dir

        # Ensure the output folder exists
        os.makedirs(output_folder, exist_ok=True)
        self.num_masks += 1

        if not os.path.exists(input_folder):
            return

        if not os.path.exists(labels_folder):
            return

        for file_name in os.listdir(input_folder):
            label_path = os.path.join(labels_folder, file_name)
            if not os.path.exists(label_path):
                continue

            # Open the label image and convert to grayscale
            label_image = Image.open(label_path).convert("L")
            label_array = np.array(label_image)

            # Initialize mask with gray (background)
            mask = np.full_like(label_array, 128, dtype=np.uint8)

            # Detect black pixels (water surface)
            mask[label_array < surface_threshold] = 0

            # Detect white pixels (fluid)
            mask[label_array > fluid_threshold] = 255

            # Save the mask
            mask_image = Image.fromarray(mask)
            output_path = os.path.join(output_folder, file_name)
            mask_image.save(output_path)
            
        print(f"Created masks folder: {output_folder}")

            
    def create_masks_from_labels(self, surface_threshold=3, fluid_threshold=253):
        """
        Creates a new folder masks/masks_{self.num_masks + 1} and stores the masks for all the images inside the labels folder.
        Also creates a new folder inputs/input_{self.num_inputs + 1} storing the original images corresponding to the labels.
        """
        labels_folder = self.labels_dir
        masks_output_folder = os.path.join(self.masks_dir, f"masks_{self.num_masks + 1}")
        inputs_output_folder = os.path.join(self.inputs_dir, f"input_{self.num_inputs + 1}")

        # Ensure the output folders exist
        os.makedirs(masks_output_folder, exist_ok=True)
        os.makedirs(inputs_output_folder, exist_ok=True)

        self.num_masks += 1
        self.num_inputs += 1

        if not os.path.exists(labels_folder):
            print(f"Error: Labels folder '{labels_folder}' does not exist.")
            return

        for file_name in os.listdir(labels_folder):
            if not file_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                continue  # Skip non-image files

            # Process the label to create a mask
            label_path = os.path.join(labels_folder, file_name)
            label_image = Image.open(label_path).convert("L")
            label_array = np.array(label_image)

            # Initialize mask with gray (background)
            mask = np.full_like(label_array, 128, dtype=np.uint8)

            # Detect black pixels (water surface)
            mask[label_array < surface_threshold] = 0

            # Detect white pixels (fluid)
            mask[label_array > fluid_threshold] = 255

            # Save the mask
            mask_image = Image.fromarray(mask)
            mask_output_path = os.path.join(masks_output_folder, file_name)
            mask_image.save(mask_output_path)

            # Extract the original image corresponding to the label
            # Parse the video and frame numbers from the file name
            try:
                parts = file_name.split("_")
                num_video = parts[1]
                num_frame = parts[2].split(".")[0]
                original_image_path = os.path.join(
                    self.frames_dir, f"frames_{num_video}", f"frame_{num_video}_{num_frame}.png"
                )

                if os.path.exists(original_image_path):
                    # Copy the original image to the new input folder
                    input_output_path = os.path.join(inputs_output_folder, file_name)
                    shutil.copy(original_image_path, input_output_path)  # Use shutil.copy instead of os.rename
                else:
                    print(f"Warning: Original image '{original_image_path}' not found. Skipping.")
            except IndexError:
                print(f"Error: Unable to parse file name '{file_name}'. Skipping.")

        print(f"Created masks folder: {masks_output_folder}")
        print(f"Created inputs folder: {inputs_output_folder}")
        
    def resize_dataset(self, index, target_size):
        # Define input and output folders for masks
        mask_input_folder = os.path.join(self.masks_dir, f"masks_{index}")
        mask_output_folder = os.path.join(self.masks_dir, f"masks_{index}_{target_size[0]}")

        # Define input and output folders for inputs
        input_input_folder = os.path.join(self.inputs_dir, f"input_{index}")
        input_output_folder = os.path.join(self.inputs_dir, f"input_{index}_{target_size[0]}")

        # Resize images in the masks folder
        if os.path.exists(mask_input_folder):
            print(f"Resizing images in {mask_input_folder} to {mask_output_folder} with size {target_size}...")
            resize_images_in_folder(mask_input_folder, mask_output_folder, target_size)
        else:
            print(f"Error: Mask folder '{mask_input_folder}' does not exist.")

        # Resize images in the inputs folder
        if os.path.exists(input_input_folder):
            print(f"Resizing images in {input_input_folder} to {input_output_folder} with size {target_size}...")
            resize_images_in_folder(input_input_folder, input_output_folder, target_size)
        else:
            print(f"Error: Input folder '{input_input_folder}' does not exist.")
    
    

class ImageMaskDataset(Dataset):
    def __init__(self, input_dir, mask_dir, img_size=(128, 128)):
        self.input_dir = input_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.input_filenames = os.listdir(input_dir)

    def __len__(self):
        return len(self.input_filenames)

    def __getitem__(self, idx):
        img_name = self.input_filenames[idx]

        # Load and preprocess image
        img_path = os.path.join(self.input_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, self.img_size) / 255.0  # Normalize to [0, 1]
        img = np.expand_dims(img, axis=0)  # Convert to 1-channel (C x H x W)

        # Load and preprocess mask
        mask_path = os.path.join(self.mask_dir, img_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, self.img_size)
        mask = np.expand_dims(mask, axis=0)

        # Map mask values to [0, 1, 2]
        mask[mask == 0] = 0  # surface
        mask[mask == 255] = 1  # fluid
        mask[mask == 128] = 2  # rest
        mask = np.clip(mask, 0, 2)

        return torch.tensor(img, dtype=torch.float32), torch.tensor(mask, dtype=torch.long)

    def save_dataset(self, save_path, train_ratio, val_ratio, test_ratio):
        """
        Splits the dataset and saves train, val, and test sets as a single .pt file.
        """
        assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1.0"

        # Shuffle dataset
        total_size = len(self)
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)
        test_size = total_size - train_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(self, [train_size, val_size, test_size])

        # Convert datasets to tensors
        def dataset_to_tensors(dataset):
            images, masks = zip(*[dataset[i] for i in range(len(dataset))])
            return torch.stack(images), torch.stack(masks)

        train_tensors = dataset_to_tensors(train_dataset)
        val_tensors = dataset_to_tensors(val_dataset)
        test_tensors = dataset_to_tensors(test_dataset)

        # Save to file
        torch.save({
            "train": train_tensors,
            "val": val_tensors,
            "test": test_tensors
        }, save_path)

        print(f"Dataset saved as {save_path}")

    

"""
MAIN
"""

videos = videos(img_size=(256, 256))
videos.extract_frames()
#videos.create_input_folder(num_frames_per_video=30)
#videos.create_masks_from_input(index=1)
videos.create_masks_from_labels()
videos.resize_dataset(index=2, target_size=(128, 128))

input_dir = "data/inputs/input_1"
mask_dir = "data/masks/masks_1"
img_size = (128, 128)
dataset = ImageMaskDataset(input_dir, mask_dir, img_size)
dataset.save_dataset(save_path="data/dataset.pt", train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)