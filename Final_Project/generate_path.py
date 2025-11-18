import cv2
import numpy as np
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

# ---------------------------------------------------------
# STEP 1: Load image and extract raw contours
# ---------------------------------------------------------

def extract_contours(image_path, threshold=128):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(image_path)

    # Binarize (invert if dark-on-light)
    _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Convert to Nx2 arrays
    contours = [c.reshape(-1, 2) for c in contours]
    return contours


# ---------------------------------------------------------
# STEP 2: Sort contours by approximate drawing order
# ---------------------------------------------------------
def sort_contours(contours):
    # Sort by the top-most point of each contour
    return sorted(contours, key=lambda c: np.min(c[:,1]))


# ---------------------------------------------------------
# STEP 3: Smooth contours using spline interpolation
# ---------------------------------------------------------

def smooth_contour(contour, smoothing=0.002):
    x = contour[:,0]
    y = contour[:,1]

    # Normalize parametric variable
    t = np.linspace(0, 1, len(contour))

    # Fit spline
    try:
        tck, _ = splprep([x, y], s=smoothing)
        x_s, y_s = splev(np.linspace(0, 1, 350), tck)  # 350 points
        return np.vstack([x_s, y_s]).T
    except:
        # Fallback: no smoothing
        return contour


# ---------------------------------------------------------
# STEP 4: Map from pixel space → 3D robot coordinates
# ---------------------------------------------------------

def pixel_to_robot(contour, scale=1.0, offset=(0,0,0)):
    ox, oy, oz = offset
    contour = np.array(contour, dtype=float)
    contour *= scale

    # Flip Y axis so origin is bottom-left instead of top-left
    contour[:,1] = -contour[:,1]

    # Add offsets
    contour3d = np.column_stack([
        contour[:,0] + ox,
        contour[:,1] + oy,
        np.full(len(contour), oz)
    ])
    return contour3d


# ---------------------------------------------------------
# STEP 5: Convert contours into robot-ready XYZ paths
# Pen-down = Z = down_height
# Pen-up = Z = up_height
# ---------------------------------------------------------

def signature_to_robot_paths(
        image_path, 
        scale=0.01, 
        offset=(0,0,0), 
        z_down=0.0, 
        z_up=0.5,
        smoothing=0.002
    ):

    raw = extract_contours(image_path)
    sorted_contours = sort_contours(raw)

    paths = []

    for contour in sorted_contours:
        smooth = smooth_contour(contour, smoothing)
        path_down = pixel_to_robot(smooth, scale, offset)

        # Pen down path
        path_down[:,2] = z_down

        # Add pen-up point at the beginning and end
        up_start = path_down[0].copy()
        up_start[2] = z_up

        up_end = path_down[-1].copy()
        up_end[2] = z_up

        full_path = np.vstack([up_start, path_down, up_end])
        paths.append(full_path)

    return paths



def animate_robot_paths(paths, interval=10, save_path=None, fade_color='lightgray'):

    # Bounds
    all_pts = np.vstack(paths)
    xmin, xmax = np.min(all_pts[:,0]), np.max(all_pts[:,0])
    ymin, ymax = np.min(all_pts[:,1]), np.max(all_pts[:,1])

    fig, ax = plt.subplots(figsize=(8,4))
    ax.set_aspect('equal')
    ax.set_xlim(xmin-0.1, xmax+0.1)
    ax.set_ylim(ymin-0.1, ymax+0.1)
    ax.set_title("Robot Signature Drawing Simulator")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # Lines: separate for pen-down and pen-up
    pen_down_lines = []
    pen_up_lines = []
    for _ in paths:
        pd_line, = ax.plot([], [], lw=2, color='black')
        pu_line, = ax.plot([], [], lw=1, color='blue', linestyle='dashed')
        pen_down_lines.append(pd_line)
        pen_up_lines.append(pu_line)

    completed_strokes = []

    # Flatten into timeline
    draw_sequence = []
    for s_idx, stroke in enumerate(paths):
        for i in range(len(stroke)-1):
            down = stroke[i,2] <= stroke[i+1,2] and stroke[i,2] != np.max(stroke[:,2])
            draw_sequence.append((s_idx, i, down))

    def update(frame):
        s_idx, p_idx, down = draw_sequence[frame]

        # Fade previous strokes
        for i, stroke in enumerate(completed_strokes):
            pen_down_lines[i].set_data(stroke[:,0], stroke[:,1])
            pen_down_lines[i].set_color(fade_color)

        # Update current stroke
        xdata = paths[s_idx][:p_idx+1,0]
        ydata = paths[s_idx][:p_idx+1,1]

        if down:
            pen_down_lines[s_idx].set_data(xdata, ydata)
        else:
            pen_up_lines[s_idx].set_data(xdata, ydata)

        # Mark stroke completed
        if p_idx == len(paths[s_idx])-2:
            completed_strokes.append(paths[s_idx])

        return pen_down_lines + pen_up_lines

    anim = FuncAnimation(fig, update,
                         frames=len(draw_sequence),
                         interval=interval,
                         blit=True,
                         repeat=False)

    if save_path:
        anim.save(save_path, fps=60)
        print(f"Animation saved to {save_path}")

    plt.show()


# ---------------------------------------------------------
# Example Usage
# ---------------------------------------------------------

if __name__ == "__main__":
    paths = signature_to_robot_paths(
        "Final_Project/test_signature_2_white_background.png", #NOTE: The image must have a white background, with no colored border
        scale=0.01,
        offset=(0, 0, 0),
        z_down=0.0,
        z_up=0.5,
        # smoothing=0.005
        smoothing = 0.005
    )

    for i, path in enumerate(paths):
        print(f"Stroke {i+1}, {len(path)} pts:")
        print(path[:5], "...\n")

    animate_robot_paths(paths, interval=5)