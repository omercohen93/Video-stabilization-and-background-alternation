import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

def movingAverage(curve, radius):
    window_size = 2 * radius + 1
    # Define the filter
    f = np.ones(window_size) / window_size
    # Add padding to the boundaries
    curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
    # Apply convolution
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    # Remove padding
    curve_smoothed = curve_smoothed[radius:-radius]
    # return smoothed curve
    return curve_smoothed


def smooth(trajectory):
    SMOOTHING_RADIUS = 150
    smoothed_trajectory = np.copy(trajectory)
    # Filter the x, y and angle curves
    for i in range(3):
        smoothed_trajectory[:, i] = movingAverage(trajectory[:, i], radius=SMOOTHING_RADIUS)
    return smoothed_trajectory


def inital_stab(Input, Output, Pbar_bool, Pbar):

    # Read input video
    cap = cv2.VideoCapture(Input)

    # Set up output video
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(Output, fourcc, fps, (w, h))

    if not Pbar_bool:
        pbar = tqdm(total=(6 * n_frames - 1), desc='Stabilization', position=0, leave=True)
    else:
        pbar = Pbar

    # Read first frame
    _, prev = cap.read()
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    pbar.update(1)
    # Pre-define transformation-store array
    transforms = np.zeros((n_frames - 1, 3), np.float32)

    for i in range(n_frames-1):
        pbar.update(1)
        # Detect feature points in previous frame
        prev_pts = cv2.goodFeaturesToTrack(prev_gray,
                                           maxCorners=1000,
                                           qualityLevel=0.01,
                                           minDistance=1,
                                           blockSize=3)

        # Read next frame
        success, curr = cap.read()
        if not success:
            break
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow (i.e. track feature points)
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

        # Filter only valid points, with lowest error
        err = err[status == 1]
        min_errors_ind = np.argpartition(err, int(len(err)*9/10), axis=0)[:int(len(err)*9/10)]
        prev_pts = prev_pts[status == 1][min_errors_ind]
        curr_pts = curr_pts[status == 1][min_errors_ind]

        # show points
        if i > 300:
            fig, ax = plt.subplots(1)
            ax.imshow(prev_gray)
            ax.scatter(prev_pts[:, 0], prev_pts[:, 1], s=10, c='r', marker="o")
            plt.show()

        # Find transformation matrix
        m = cv2.estimateRigidTransform(prev_pts, curr_pts, fullAffine=False)  # will only work with OpenCV-3 or less
        # Extract traslation
        dx = m[0, 2]
        dy = m[1, 2]
        # Extract rotation angle
        da = np.arctan2(m[1, 0], m[0, 0])
        # Store transformation
        transforms[i] = [dx, dy, da]

        # Move to next frame
        prev_gray = curr_gray


    # Compute trajectory using cumulative sum of transformations
    trajectory = np.cumsum(transforms, axis=0)
    # Create variable to store smoothed trajectory
    smoothed_trajectory = smooth(trajectory)
    # Calculate difference in smoothed_trajectory and trajectory
    difference = smoothed_trajectory - trajectory
    # Calculate newer transformation array
    transforms_smooth = transforms + difference

    # Reset stream to first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Write n_frames-1 transformed frames
    for i in range(-1, n_frames-1):
        pbar.update(1)
        # Read next frame
        success, frame = cap.read()
        if not success:
            break
        if i == -1:
            i = 0
        # Extract transformations from the new transformation array
        dx = transforms_smooth[i, 0]
        dy = transforms_smooth[i, 1]
        da = transforms_smooth[i, 2]

        # Reconstruct transformation matrix accordingly to new values
        m = np.zeros((2, 3), np.float32)
        m[0, 0] = np.cos(da)
        m[0, 1] = -np.sin(da)
        m[1, 0] = np.sin(da)
        m[1, 1] = np.cos(da)
        m[0, 2] = dx
        m[1, 2] = dy

        # Apply affine wrapping to the given frame
        frame_stabilized = cv2.warpAffine(frame, m, (w, h))

        # Write the frame to the file

        out.write(frame_stabilized)

    # Release video
    cap.release()
    out.release()
    # Close windows
    cv2.destroyAllWindows()
    return pbar


def median_trajectory(trajectory):
    med_trajectory = np.copy(trajectory)
    # Filter the x, y and angle curves
    for i in range(2):
        med_trajectory[:, i] = np.median(trajectory[:, i], keepdims=True)
    return med_trajectory

def center_vid(Input, Output, Pbar):
    pbar = Pbar

    snip = 60

    # Read input video
    cap = cv2.VideoCapture(Input)

    # Set up output video
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(Output, fourcc, fps, (w, h))

    # Read first frame
    _, prev = cap.read()
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    pbar.update(1)
    # Pre-define transformation-store array
    transforms = np.zeros((n_frames - 1, 2), np.float32)

    for i in range(n_frames-1):
        pbar.update(1)
        # Detect feature points in previous frame

        prev_pts = cv2.goodFeaturesToTrack(prev_gray,
                                           maxCorners=1000,
                                           qualityLevel=0.01,
                                           minDistance=1,
                                           blockSize=3)
        # Read next frame
        success, curr = cap.read()
        if not success:
            break

            # Convert to grayscale
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow (i.e. track feature points)
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
        # Sanity check
        assert prev_pts.shape == curr_pts.shape

        # Filter only valid points
        err = err[status == 1]
        min_errors_ind = np.argpartition(err, int(len(err)*9/10), axis=0)[:int(len(err)*9/10)]

        prev_pts = prev_pts[status == 1][min_errors_ind]
        curr_pts = curr_pts[status == 1][min_errors_ind]

        if i > 300:
            fig, ax = plt.subplots(1)
            ax.imshow(prev_gray)
            ax.scatter(prev_pts[:, 0], prev_pts[:, 1], s=10, c='r', marker="o")
            plt.show()

        # Find transformation matrix
        x = curr_pts[:, 0] - prev_pts[:, 0]
        y = curr_pts[:, 1] - prev_pts[:, 1]

        # Extract traslation
        dx = np.median(x)
        dy = np.median(y)

        # Store transformation
        transforms[i] = [dx, dy]

        # Move to next frame
        prev_gray = curr_gray

    # Compute trajectory using cumulative sum of transformations
    trajectory = np.cumsum(transforms, axis=0)

    # Create variable to store smoothed trajectory
    smoothed_trajectory = median_trajectory(trajectory)

    # Calculate difference in smoothed_trajectory and trajectory
    difference = smoothed_trajectory - trajectory

    # Calculate newer transformation array
    transforms_smooth = transforms + difference

    # Reset stream to first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Write n_frames-1 transformed frames
    for i in range(-1, n_frames-1):
        pbar.update(1)
        # Read next frame
        success, frame = cap.read()
        if not success:
            break

        if i == -1:
            i = 0

        # Extract transformations from the new transformation array
        dx = transforms_smooth[i, 0]
        dy = transforms_smooth[i, 1]
        # Reconstruct transformation matrix accordingly to new values
        m = np.zeros((2, 3), np.float32)
        m[0, 0] = 1
        m[1, 1] = 1
        m[0, 2] = dx
        m[1, 2] = dy

        # Apply affine wrapping to the given frame
        frame_stabilized = cv2.warpAffine(frame, m, (w, h))

        # Write the frame to the file
        if snip > 0:
            frame_stabilized = frame_stabilized[snip:-snip, snip:-snip]
            frame_stabilized = np.pad(frame_stabilized, ((snip, snip), (snip, snip), (0, 0)), mode='constant')
            frame_stabilized = np.array(frame_stabilized, dtype='uint8')
        out.write(frame_stabilized)

    # Release video
    cap.release()
    out.release()
    # Close windows
    cv2.destroyAllWindows()
    pbar.close()


