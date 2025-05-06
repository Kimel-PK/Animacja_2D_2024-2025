import cv2
import numpy as np
from PIL import Image, ImageDraw
import random
import math
import os
import imageio.v2 as imageio
import colorsys
import shutil

# === Configuration ===
WIDTH, HEIGHT = 600, 600
NUM_SHAPES = 60
NUM_FRAMES = 1800
SHAPE_TYPES = ['circle', 'hollow_circle', 'rectangle', 'hollow_rectangle', 'triangle', 'hollow_triangle']
OUTPUT_DIR = "frames"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def random_saturated_color():
    # Random hue value (0-360)
    hue = random.random()  # Random float between 0.0 and 1.0
    # Convert to HSV (Hue, Saturation, Value)
    rgb = colorsys.hsv_to_rgb(hue, 1, 1)  # Saturation and Value are set to 1 (full saturation and brightness)
    # Convert RGB to 0-255 range and return as a tuple
    return tuple(int(c * 255) for c in rgb)

# === Shape Class ===
class Shape:
    def __init__(self):
        self.type = random.choice(SHAPE_TYPES)
        self.size = random.randint(30, 120)
        self.position = np.array([
            random.uniform(self.size, WIDTH),
            random.uniform(self.size, HEIGHT)
        ])
        self.velocity = np.random.uniform(-1, 1, size=2)
        self.rotation = random.uniform(0, 360)
        self.angular_velocity = random.uniform(-2, 2)
        self.color = random_saturated_color()

    def update(self):
        # Move shape
        self.position += self.velocity
        self.rotation += self.angular_velocity

        # Bounce off edges
        if self.position[0] < 0 or self.position[0] > WIDTH:
            self.velocity[0] *= -1
            self.position[0] = np.clip(self.position[0], 0, WIDTH)
        if self.position[1] < 0 or self.position[1] > HEIGHT:
            self.velocity[1] *= -1
            self.position[1] = np.clip(self.position[1], 0, HEIGHT)

    def draw(self, draw_obj):
        x, y = self.position
        angle = math.radians(self.rotation)

        if self.type == 'circle':
            bbox = [x - self.size, y - self.size, x + self.size, y + self.size]
            draw_obj.ellipse(bbox, fill=self.color)
            
        elif self.type == 'hollow_circle':
            bbox = [x - self.size, y - self.size, x + self.size, y + self.size]
            # Draw the outline of the circle with a width (e.g., 5px)
            draw_obj.ellipse(bbox, outline=self.color, width=8)

        elif self.type == 'rectangle':
            half = self.size / 2
            corners = [(-half, -half), (half, -half), (half, half), (-half, half)]
            rotated = [(
                x + dx * math.cos(angle) - dy * math.sin(angle),
                y + dx * math.sin(angle) + dy * math.cos(angle)
            ) for dx, dy in corners]
            draw_obj.polygon(rotated, fill=self.color)
            
        elif self.type == 'hollow_rectangle':
            half = self.size / 2
            corners = [(-half, -half), (half, -half), (half, half), (-half, half)]
            
            # Rotate corners around the center (x, y)
            rotated = [(
                x + dx * math.cos(angle) - dy * math.sin(angle),
                y + dx * math.sin(angle) + dy * math.cos(angle)
            ) for dx, dy in corners]
            
            # Draw the edges with width 5px
            # You can draw lines between the consecutive corners to form a hollow rectangle
            for i in range(4):
                start_point = rotated[i]
                end_point = rotated[(i + 1) % 4]  # Ensures the last point connects to the first
                draw_obj.line([start_point, end_point], fill=self.color, width=8)

        elif self.type == 'triangle':
            half = self.size / 2
            points = [(-half, half), (0, -half), (half, half)]
            rotated = [(
                x + px * math.cos(angle) - py * math.sin(angle),
                y + px * math.sin(angle) + py * math.cos(angle)
            ) for px, py in points]
            draw_obj.polygon(rotated, fill=self.color)
            
        elif self.type == 'hollow_triangle':
            half = self.size / 2
            points = [(-half, half), (0, -half), (half, half)]
            
            # Rotate the points around the center (x, y)
            rotated = [(
                x + px * math.cos(angle) - py * math.sin(angle),
                y + px * math.sin(angle) + py * math.cos(angle)
            ) for px, py in points]
            
            # Draw the outline of the triangle (hollow) with a width (e.g., 5px)
            draw_obj.polygon(rotated, outline=self.color, width=8)

def kaleidoscopify(framePath):
    # arguments
    invert = "no"     # invert mask; yes or no
    rotate = 0        # rotate composite; 0, 90, 180, 270

    # read image (must be square)
    img = cv2.imread(framePath)
    ht, wd = img.shape[:2]

    # transpose the image
    imgt = cv2.transpose(img)

    # create diagonal bi-tonal mask
    mask = np.zeros((ht,wd), dtype=np.uint8)
    points = np.array( [[ [0,0], [wd,0], [wd,ht] ]] )
    cv2.fillPoly(mask, points, 255)
    if invert == "yes":
        mask = cv2.bitwise_not(mask)

    # composite img and imgt using mask
    compA = cv2.bitwise_and(imgt, imgt, mask=mask)
    compB = cv2.bitwise_and(img, img, mask=255-mask)
    comp = cv2.add(compA, compB)

    # rotate composite
    if rotate == 90:
        comp = cv2.rotate(comp,cv2.ROTATE_90_CLOCKWISE)
    elif rotate == 180:
        comp = cv2.rotate(comp,cv2.ROTATE_180)
    elif rotate == 270:
        comp = cv2.rotate(comp,cv2.ROTATE_90_COUNTERCLOCKWISE)

    # mirror (flip) horizontally
    mirror = cv2.flip(comp, 1)

    # concatenate horizontally
    top = np.hstack((comp, mirror))

    # mirror (flip) vertically
    bottom = cv2.flip(top, 0)

    # concatenate vertically
    kaleidoscope_big = np.vstack((top, bottom))

    # resize
    kaleidoscope = cv2.resize(kaleidoscope_big, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

    # save results
    cv2.imwrite(framePath, kaleidoscope)

def images_to_video(input_folder, output_file="output.mp4", fps=30):
    # Get all image file paths, sorted by name
    image_files = sorted([
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    if not image_files:
        raise ValueError("No images found in the input folder.")

    # Read the first image to get dimensions
    first_frame = cv2.imread(image_files[0])
    height, width, _ = first_frame.shape

    # Define the video codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    for image_path in image_files:
        frame = cv2.imread(image_path)
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        out.write(frame)

    out.release()
    print(f"Video saved as {output_file}")

# === Main Loop ===
shapes = [Shape() for _ in range(NUM_SHAPES)]

os.makedirs('frames', exist_ok=True)

for frame_idx in range(NUM_FRAMES):
    img = Image.new('RGB', (WIDTH, HEIGHT), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    for shape in shapes:
        shape.draw(draw)
        shape.update()

    img.save(os.path.join(OUTPUT_DIR, f'frame_{frame_idx:04d}.png'))
    kaleidoscopify(os.path.join(OUTPUT_DIR, f'frame_{frame_idx:04d}.png'))

images_to_video("frames", "kaleidoscope.mp4", fps=60)

shutil.rmtree('frames')

print("Done generating frames.")
