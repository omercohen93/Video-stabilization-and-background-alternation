import cv2
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
from tqdm import tqdm

def compNormHist(I, S):
    x, y = S[0:2]
    w, h = S[2:4]
    # avoid nan argument in hist
    if x-w < 0:
        x = w
    if x+w > I.shape[1]-1:
        x = I.shape[1]-1-w
    if y-h < 0:
        y = h
    if y+h > I.shape[0]-1:
        y = I.shape[0]-1-h

    I_temp = I[y-h: y+h+1, x-w: x+w+1]
    hist = cv2.calcHist([I_temp], [0, 1, 2], None, [16, 16, 16], [0, 256, 0, 256, 0, 256])
    hist = hist.reshape(hist.size)/np.sum(hist)
    return hist

def predictParticles(S_next_tag):
    dx = S_next_tag[-2,:]
    dy = S_next_tag[-1,:]
    S_next = np.array(S_next_tag)
    # predict x and y
    S_next[0, :] += np.array(dx + np.matrix.round(np.random.uniform(-5, 5, S_next[0, :].shape)), dtype=int)
    S_next[1, :] += np.array(dy + np.matrix.round(np.random.uniform(-5, 5, S_next[0, :].shape)), dtype=int)
    # set dx and dy as differences between original x,y and predicted x,y of the same S vector
    S_next[-2:, :] = S_next[0:2, :] - S_next_tag[0:2, :]
    return S_next

def compBatDist(p, q):
    return np.exp(20*np.sum((p * q)**0.5))

def sampleParticles(S_prev, C):
    S_next_tag = S_prev
    for n in range(S_prev.shape[1]):
        r = np.random.random()
        j = -np.size(C[C >= r])
        S_next_tag[:, n] = S_prev[:, j]
    return S_next_tag

def showParticles(I, S, W):
    rect_hight = 2*S[3, 0] + 1
    rect_width = 2*S[2, 0] + 1
    x_avg = int(round(sum((x * weight for x, weight in zip(S[0, :], W))))) - S[2, 0]
    y_avg = int(round(sum((y * weight for y, weight in zip(S[1, :], W))))) - S[3, 0]

    # adding rectangles (first image is original template
    frame_out = cv2.rectangle(I, (x_avg, y_avg), (x_avg+rect_width, y_avg+rect_hight), (0, 255, 0))
    return frame_out

def comp_weight(S, q, I):  # computes the Weight and CMD
    p_range = [compNormHist(I, S[:, i]) for i in range(S.shape[1])]
    W = [compBatDist(p, q) for p in p_range]
    W /= sum(W)
    C = np.array([sum(W[0:i+1]) for i in range(W.size)])
    return W, C


def track_object(input_vid, vid_with_rect):
    # Read input video
    cap = cv2.VideoCapture(input_vid)

    # Set up output video
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(vid_with_rect, fourcc, fps, (w, h))
    pbar = tqdm(total=(n_frames - 1), desc='track_object', position=0, leave=True)
    # Read first frame
    _, frame = cap.read()
    #pbar.update(1)
    # SET NUMBER OF PARTICLES
    N = 100

    # Initial Settings
    s_initial = [158,    # x center
                 500,    # y center
                 50,    # half width
                 220,    # half height
                 0,    # velocity x
                 0]    # velocity y

    # CREATE INITIAL PARTICLE MATRIX 'S' (SIZE 6xN)
    S = predictParticles(np.matlib.repmat(s_initial, N, 1).T)

    # COMPUTE NORMALIZED HISTOGRAM for first frame
    q = compNormHist(frame, s_initial)
    # COMPUTE NORMALIZED WEIGHTS (W) AND PREDICTOR CDFS (C)
    W, C = comp_weight(S, q, frame)
    # put rect on first image
    frame = showParticles(frame, S, W)
    frame = np.uint8(frame)
    out.write(frame)

    # MAIN TRACKING LOOP
    for i in range(1, n_frames):
        pbar.update(1)
        S_prev = S
        # LOAD NEW IMAGE FRAME
        success, frame = cap.read()
        if not success:
            break
        # SAMPLE THE CURRENT PARTICLE FILTERS
        S_next_tag = sampleParticles(S_prev, C)

        # PREDICT THE NEXT PARTICLE FILTERS (YOU MAY ADD NOISE
        S = predictParticles(S_next_tag)

        # COMPUTE NORMALIZED WEIGHTS (W) AND PREDICTOR CDFS (C)
        W, C = comp_weight(S, q, frame)

        # CREATE DETECTOR rect
        frame = showParticles(frame, S, W)
        frame = np.uint8(frame)
        out.write(frame)
    pbar.close()
