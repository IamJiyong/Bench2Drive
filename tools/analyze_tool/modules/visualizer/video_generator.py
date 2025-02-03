import os
import cv2
import argparse
from tqdm import tqdm

class VideoGenerator:
    def __init__(self, output_root, fps=15):
        self.fps = fps
        self.output_root = output_root

    def generate_and_save(self, images, scenario, camera):
        """
        Generate and save a video from a list of images.
        Args:
            images (list): A list of image frames (numpy arrays) to be converted to a video.
            meta (dict): Metadata for the video.
                'scenario': Name of the scenario
                'camera': Name of the camera
        Returns:
            bool: True if video was saved successfully, False otherwise.
        """
        first_image = images[0]
        height, width, _ = first_image.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # Ensure scenario output directory exists
        output_folder = os.path.join(self.output_root, 'videos', scenario)
        os.makedirs(output_folder, exist_ok=True)

        output_video = os.path.join(output_folder, f"{camera}.mp4")
        video = cv2.VideoWriter(output_video, fourcc, self.fps, (width, height))

        for img in tqdm(images, desc=f"Generating {output_video}"):
            video.write(img)

        video.release()
        print(f"✅ Video saved: {output_video}")
        return True


def load(image_dir):
    """
    Loads images from folders and returns them.
    """
    assert os.path.exists(image_dir), f"❌ Error: Folder '{image_dir}' does not exist."

    image_files = sorted(
        [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".png")]
    )

    images = []
    for image_path in image_files:
        img = cv2.imread(image_path)
        if img is None:
            print(f"⚠️ Warning: Skipping unreadable image '{image_path}'")
            continue
        images.append(img)
    
    return images


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate videos from images.")
    parser.add_argument("--eval_dir", nargs='+', required=True, help="List of image folder groups.")
    parser.add_argument("--cameras", required=True, nargs='+', help="Root directory for saving output videos.")
    parser.add_argument("--fps", type=int, default=15, help="Frames per second for the output video")
    
    args = parser.parse_args()
    
    for eval_dir in args.eval_dir:
        assert os.path.exists(eval_dir), f"❌ Error: Folder '{eval_dir}' does not exist."

        video_generator = VideoGenerator(eval_dir, args.fps)
        
        # if scenario starts with 'RouteScenario', then it is a scenario folder
        scenario_names = [d for d in os.listdir(eval_dir) if d.startswith("RouteScenario")]

        for scenario in scenario_names:
            scenario_dir = os.path.join(eval_dir, scenario)
            for camera in args.cameras:
                camera_dir = os.path.join(scenario_dir, camera)
                images = load(camera_dir)
                video_generator.generate_and_save(images, scenario, camera)
    