import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

def back_sub(Stab_vid, back_sub1, back_sub2, t_window, hsv_dim):
    # Read input video
    cap = cv2.VideoCapture(Stab_vid)

    # Set up output video
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(back_sub1, fourcc, fps, (w, h), isColor=False)
    out2 = cv2.VideoWriter(back_sub2, fourcc, fps, (w, h), isColor=False)

    pbar = tqdm(total=(2 * n_frames - 1), desc='back_sub', position=0, leave=True)

    ######################################### build background list #########################################
    time_window_size = t_window
    threshold1 = 20
    threshold2 = 55
    time_jump = 5

    images_in_window = []   # list of images to median
    background_list = []   # list of backgrounds
    for i in range(n_frames):
        pbar.update(1)
        # Read next frame
        success, frame = cap.read()
        if not success:
            break
        # get chosen dim in hsv image, and add to list
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, hsv_dim]
        images_in_window.append(frame)

        # when list is in proper size, make median of frames
        if len(images_in_window) > time_window_size-1:
            images_in_window = images_in_window[1:]
        # added time jump, if shorted runtime is desired. works well with stable videos
        if time_window_size-1 <= i and i % time_jump == 0:
            back_im = np.uint8(np.median(np.array(images_in_window), axis=0))
            background_list.append(back_im)

    cap.release()
    cv2.destroyAllWindows()

    ######################################### background substrect #########################################

    cap = cv2.VideoCapture(Stab_vid)

    # create index indecating what background image to choose from list
    back_index = 0

    for i in range(n_frames):
        pbar.update(1)

        # Read next frame, get chosen dim in hsv image
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, hsv_dim]

        # propagation in back_im list index
        if int(time_window_size / 2) < i < n_frames - int(time_window_size / 2)-time_jump and i % time_jump == 0:
            back_index += 1
        back_im = np.array(background_list[back_index], dtype='uint8')

        # substract
        frame_mat = np.array(frame, dtype='int_')
        back_im_mat = np.array(back_im, dtype='int_')
        frame_out = np.array(np.abs(frame_mat - back_im_mat), dtype='uint8')
        frame_gray = frame_out

        # bin thres image
        ret, frame_out1 = cv2.threshold(frame_gray, threshold1, 255, cv2.THRESH_BINARY)
        ret, frame_out2 = cv2.threshold(frame_gray, threshold2, 255, cv2.THRESH_BINARY)

        out.write(frame_out1)
        out2.write(frame_out2)
    # Release video
    cap.release()
    out.release()
    out2.release()
    cv2.destroyAllWindows()
    pbar.close()

def sum_vid(input1, input2, input3, output):
    # Read input video
    cap1 = cv2.VideoCapture(input1)
    cap2 = cv2.VideoCapture(input2)
    cap3 = cv2.VideoCapture(input3)

    # set output
    n_frames = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap2.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    pbar = tqdm(total=(n_frames - 1), desc='sum_vid', position=0, leave=True)
    out = cv2.VideoWriter(output, fourcc, fps, (w, h), isColor=False)

    for i in range(n_frames):
        pbar.update(1)
        # Read next frame
        success1, frame1 = cap1.read()
        success2, frame2 = cap2.read()
        success3, frame3 = cap3.read()
        if not success1 or not success2:
            break
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        frame3 = cv2.cvtColor(frame3, cv2.COLOR_BGR2GRAY)
        ret, frame1 = cv2.threshold(frame1, 50, 255, cv2.THRESH_BINARY)
        ret, frame2 = cv2.threshold(frame2, 50, 255, cv2.THRESH_BINARY)
        ret, frame3 = cv2.threshold(frame3, 50, 255, cv2.THRESH_BINARY)
        frame2 = np.array(frame2, dtype='bool')
        frame3 = np.array(frame3, dtype='bool')
        frame1[frame2] = 255
        frame1[frame3] = 255
        frame1 = np.array(frame1, dtype='uint8')

        out.write(frame1)
    # Release video
    cap1.release()
    cap2.release()
    out.release()
    pbar.close()
    # Close windows
    cv2.destroyAllWindows()


def region_of_interest(Input, output):
    # Read input video
    cap = cv2.VideoCapture(Input)

    # Set up output video
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output, fourcc, fps, (w, h), isColor=False)
    pbar = tqdm(total=(n_frames - 1), desc='region_of_interest', position=0, leave=True)

    threshold1 = 250

    for i in range(n_frames):
        pbar.update(1)
        # Read next frame
        success, frame = cap.read()
        if not success:
            break
        frame = np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), dtype='float32')
        ret, frame = cv2.threshold(frame, 50, 255, cv2.THRESH_BINARY)

        # get  rect of largest contour
        frame1 = cv2.erode(frame, np.ones((8, 8)))
        frame1 = cv2.dilate(frame1, np.ones((50, 50)))
        frame1 = np.uint8(frame1)
        _, Contour, _ = cv2.findContours(frame1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, width, hight = cv2.boundingRect(max(Contour, key=cv2.contourArea))
        # set frame within rect boundries
        frame_out = np.zeros_like(frame)
        frame_out[y:y + hight, x:x + width] = frame[y:y + hight, x:x + width]
        frame_out = np.uint8(frame_out)

        # get smaller mask within rect of object region of interest
        for j in range(2):
            frame_out = cv2.filter2D(frame_out, -1, np.ones((5, 5), np.float32) / 25)
            ret, frame_out = cv2.threshold(frame_out, threshold1, 255, cv2.THRESH_BINARY)

        frame_out = cv2.filter2D(frame_out, -1, np.ones((40, 40), np.float32))
        ret, frame_out = cv2.threshold(frame_out, 1, 255, cv2.THRESH_BINARY)

        Bool = np.array(frame_out == np.zeros_like(frame_out), dtype='bool')
        frame[Bool] = 0

        frame = np.uint8(frame)
        out.write(frame)

    # Release video
    cap.release()
    out.release()
    pbar.close()
    # Close windows
    cv2.destroyAllWindows()












def dominant_hues(Input1, Input2, Output):
    # Read input video
    cap = cv2.VideoCapture(Input1)
    cap2 = cv2.VideoCapture(Input2)

    # set output
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(Output, fourcc, fps, (w, h), isColor=False)
    pbar = tqdm(total=(2 * n_frames - 1), desc='dominant_hues', position=0, leave=True)
    # init list of hues
    hist_list = np.zeros((180, 1))

    # define ratio of hues to neglect, only 3 will be left in list
    reduce = 0.98
    # spectrum around hue to keep (from each side)
    space = 6
    ######################################### build hue list #########################################
    for i in range(n_frames):

        pbar.update()
        # Read next frame
        success, frame = cap.read()
        success2, frame2 = cap2.read()
        if not success or not success2:
            break
        # set hue image
        frame_h = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 0]
        # define region of interest
        frame_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        ret, frame_gray = cv2.threshold(frame_gray, 10, 255, cv2.THRESH_BINARY)
        Bool = np.array(frame_gray, dtype='bool')
        frame_h = frame_h[Bool]

        # calc hue histogram
        hist = cv2.calcHist([frame_h], [0], None, [180], [1, 180])
        # emphasize repeated pixels
        hist_list += 1e20 * hist

    # get most popular hues
    hue_list = np.argpartition(hist_list.copy(), int(180 * reduce), axis=0)[int(180 * reduce):]

    # Reset stream to first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)

    ###################################### get mask of dominant hues spectrom ######################################
    for i in range(n_frames):
        pbar.update()
        # Read next frame
        success, frame = cap.read()
        success2, frame2 = cap2.read()
        if not success or not success2:
            break
        # set hue image
        frame_h = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 0]
        # define region of interest
        frame_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        frame_gray_copy = np.copy(frame_gray)

        # neglect hues out of spectrum
        for j in hue_list:
            lower = np.array([j - space])
            uper = np.array([j + space])
            mask = np.array(cv2.inRange(frame_h, lower, uper), dtype='bool')

            frame_gray_copy[mask] = 0
        frame_gray[frame_gray_copy != 0] = 0

        # fill holes
        frame_gray = cv2.morphologyEx(frame_gray, cv2.MORPH_CLOSE, np.ones((10, 10)))

        out.write(frame_gray)

    # Release video
    cap.release()
    cap2.release()
    out.release()
    pbar.close()
    # Close windows
    cv2.destroyAllWindows()

def fine_tune(Input, output):
    # Read input video
    cap = cv2.VideoCapture('center_mas_rect_S&V_filtered.avi')

    # Set up output video
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('final_bin_mask.avi', fourcc, fps, (w, h), isColor=False)
    pbar = tqdm(total=(n_frames - 1), desc='fine_tune', position=0, leave=True)
    for i in range(n_frames):
        pbar.update(1)
        # Read next frame
        success1, frame1 = cap.read()

        if not success1:
            break
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        ret, frame1 = cv2.threshold(frame1, 100, 255, cv2.THRESH_BINARY)
        frame1 = cv2.erode(frame1, np.ones((5, 5)), iterations=1)
        frame1 = cv2.morphologyEx(frame1, cv2.MORPH_CLOSE, np.ones((10, 10)))
        out.write(frame1)
    # Release video
    cap.release()
    out.release()

    # Close windows
    cv2.destroyAllWindows()


def get_object(mask, stabvid, bin_out, color_out):
    # Read input video
    cap = cv2.VideoCapture(mask)
    cap2 = cv2.VideoCapture(stabvid)

    # Set up output video
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fourcc2 = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(bin_out, fourcc, fps, (w, h), isColor=False)
    out2 = cv2.VideoWriter(color_out, fourcc2, fps, (w, h))

    pbar = tqdm(total=(n_frames - 1), desc='get_object', position=0, leave=True)
    for i in range(n_frames):
        pbar.update(1)
        # Read next frame
        success1, frame_bin = cap.read()
        success2, frame_col = cap2.read()
        if not success1 or not success2:
            break
        frame_bin = cv2.cvtColor(frame_bin, cv2.COLOR_BGR2GRAY)
        ret, frame_bin = cv2.threshold(frame_bin, 100, 255, cv2.THRESH_BINARY)

        # lose noise, fill holes
        frame_bin = cv2.erode(frame_bin, np.ones((5, 5)), iterations=1)
        frame_bin = cv2.morphologyEx(frame_bin, cv2.MORPH_CLOSE, np.ones((25, 25)))
        Bool = np.array(frame_bin == np.zeros_like(frame_bin), dtype='bool')

        frame_col[:, :, 0][Bool] = 0
        frame_col[:, :, 1][Bool] = 0
        frame_col[:, :, 2][Bool] = 0
        frame_col = np.array(frame_col, dtype='uint8')

        out.write(frame_bin)
        out2.write(frame_col)

    # Release video
    cap.release()
    out.release()

    # Close windows
    cv2.destroyAllWindows()
    pbar.close()
    pbar.close()

