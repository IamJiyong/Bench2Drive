import os
import cv2
from tqdm import tqdm


def create_video(images_folder, output_video, fps=15):
    """
    Creates a video from images stored in a specific folder.

    Args:
        images_folder (str): Path to the folder containing images.
        output_video (str): Path to the output video file.
        fps (int, optional): Frames per second of the output video. Default is 15.
    """
    if not os.path.exists(images_folder):
        print(f"❌ Error: Folder '{images_folder}' does not exist.")
        return

    image_files = sorted(
        [f for f in os.listdir(images_folder) if f.endswith((".jpg", ".png"))]
    )

    if not image_files:
        print(f"⚠️ Warning: No images found in '{images_folder}'. Skipping...")
        return

    first_image_path = os.path.join(images_folder, image_files[0])
    frame = cv2.imread(first_image_path)

    if frame is None:
        print(f"❌ Error: Could not read the first image '{first_image_path}'.")
        return

    height, width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Ensure output directory exists
    output_folder = os.path.dirname(output_video)
    os.makedirs(output_folder, exist_ok=True)  # Ensure the output folder exists

    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    print(f"🎥 Generating {output_video}...")

    for image_name in tqdm(image_files, desc=f"Generating {output_video}"):
        image_path = os.path.join(images_folder, str(image_name))
        img = cv2.imread(image_path)

        if img is None:
            print(f"⚠️ Warning: Skipping unreadable image '{image_path}'")
            continue

        video.write(img)

    video.release()
    print(f"✅ Video saved: {output_video}")


def main():
    """
    Main function to generate videos for each model type, scenario, and camera type.
    """
    #-------------------------Config Settings-------------------------#
    model_types = ["tcp"]   # model_types = ["tcp", "uniad", "vad"] 
 
    base_root = "/home/ysh/jiyong/b2d_carla/Bench2Drive"  # Base directory

    camera_types = ["rgb_front", "bev"]     # Camera types to process
    #-----------------------------------------------------------------#


    for model_type in model_types:
        base_folder = os.path.join(base_root, f"eval_v1_output_{model_type}")

        if not os.path.exists(base_folder):
            print(f"❌ Error: Base folder '{base_folder}' not found. Skipping '{model_type}' model.")
            continue

        # Get all scenario directories
        scenarios = sorted(
            [d for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d))]
        )

        for scenario in scenarios:
            scenario_path = os.path.join(base_folder, scenario)
            print(f"\n📂 Processing scenario: {scenario} (Model: {model_type})")

            for camera_type in camera_types:
                camera_path = os.path.join(scenario_path, camera_type)

                if not os.path.exists(camera_path):
                    print(f"⚠️ Warning: Camera folder '{camera_path}' not found. Skipping...")
                    continue

                # Create the output directory
                output_video_dir = os.path.join(scenario_path, "output_video")
                
                # Ensure output_video folder is created
                os.makedirs(output_video_dir, exist_ok=True)
                print(f"📁 Ensuring output directory exists: {output_video_dir}")

                # Set output video path
                output_video_path = os.path.join(output_video_dir, f"{camera_type}.mp4")
                print(f"🎥 Creating video for {camera_type} in {scenario} (Model: {model_type})...")

                create_video(camera_path, output_video_path)


if __name__ == "__main__":
    main()
