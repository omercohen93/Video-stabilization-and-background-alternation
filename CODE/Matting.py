import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import time
from tqdm import tqdm

def get_alpha(stab_vid, bin_mask, alpha_out):
    # Read input video
    cap = cv2.VideoCapture(bin_mask)
    cap2 = cv2.VideoCapture(stab_vid)

    # set output
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(alpha_out, fourcc, fps, (w, h), isColor=False)
    pbar = tqdm(total=(n_frames - 1), desc='get_alpha', position=0, leave=True)
    # set divided regions for kde
    devide = 2
    # parameters to change
    sigma_bg = 0.007
    sigma_fg = 0.005
    R = 0.01
    top_alpha = 0.6
    bot_alpha = 0
    stat_data_radius = 60
    inflate = 10
    show_alpha = False
    show_w = False
    show_geo_dist = False
    show_pmap = False

    for i in range(n_frames):
        pbar.update(1)
        # Read next frame
        success1, frame_bin = cap.read()
        success2, frame_color = cap2.read()
        if not success1 or not success2:
            break

        # fix bin_frame
        frame_bin = cv2.cvtColor(frame_bin, cv2.COLOR_BGR2GRAY)
        ret, frame_bin = cv2.threshold(frame_bin, 100, 255, cv2.THRESH_BINARY)

        # defin thick and thin masks for FG and BG definition
        thick_frame_bin = np.copy(frame_bin)
        thin_frame_bin = np.copy(frame_bin)

        # erode to avoid unwanted pixels in FG
        thin_frame_bin = cv2.filter2D(thin_frame_bin, -1, np.ones((16, 16)) / 256)
        ret, thin_frame_bin = cv2.threshold(thin_frame_bin, 235, 255, cv2.THRESH_BINARY)

        # dialat for thicker object
        thick_frame_bin = cv2.dilate(thick_frame_bin, np.ones((3,3)), iterations=4)
        thick_frame_bin = cv2.morphologyEx(thick_frame_bin, cv2.MORPH_CLOSE, np.ones((25,25)))

        # def forground and bacground
        BG = np.ones_like(thick_frame_bin)*255
        BG[thick_frame_bin] = 0
        FG = thin_frame_bin

        # work with S layer
        frame_S = cv2.cvtColor(frame_color, cv2.COLOR_BGR2HSV)[:, :, 1]

        # get object location by finding contour
        _, Contour, _ = cv2.findContours(thick_frame_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, width, hight = cv2.boundingRect(max(Contour, key=cv2.contourArea))

        # wider rect- the code gets wider rect
        if True:
            if x-inflate < 0:
                x = 0
            else:
                x -= inflate
            if x + width + 2*inflate >= w:
                width = w - x - 1
            else:
                width += 2*inflate

        cv2.rectangle(thin_frame_bin, (x,y), (x+width, y+hight), 255, thickness=10)

        # calc cropedGeodesic distance map
        mask_S = np.zeros_like(frame_S)
        mask_S[thick_frame_bin > 0] = frame_S[thick_frame_bin > 0]
        dist_map_FG = cv2.distanceTransform(mask_S, cv2.DIST_LABEL_PIXEL, 0)[y:y+hight, x:x+width]
        dist_map_BG = 255 - dist_map_FG

        if show_geo_dist:
            fig = plt.figure()
            a = fig.add_subplot(1, 2, 1)
            imgplot = plt.imshow(dist_map_FG)
            a.set_title('dist_map_FG')
            a = fig.add_subplot(1, 2, 2)
            imgplot = plt.imshow(dist_map_BG)
            a.set_title('dist_map_BG')
            plt.show()

        # get values in rect
        crop_frame_S = frame_S[y:y + hight, x:x + width]
        S_values = np.unique(crop_frame_S)

        if i % stat_data_radius == 0:
            for j in range(devide):
                h_cur = int(j * hight / devide)
                h_next = int((j+1) * hight / devide)
                w_cur = int(j * width / devide)
                w_next = int((j + 1) * width / devide)
                # crop BG and FG
                sub_crop_frame_S = frame_S[y + h_cur:y + h_next, x + w_cur:x + w_next]
                crop_BG = sub_crop_frame_S[BG[y + h_cur:y + h_next, x + w_cur:x + w_next] == 255]
                crop_FG = sub_crop_frame_S[FG[y + h_cur:y + h_next, x + w_cur:x + w_next] == 255]


                # calc conditional probability
                BG_kde = gaussian_kde(crop_BG, sigma_bg)
                sub_p_given_BG = BG_kde(S_values)
                FG_kde = gaussian_kde(crop_FG, sigma_fg)
                sub_p_given_FG = FG_kde(S_values)

                if j == 0:
                    p_given_FG = sub_p_given_FG / devide
                    p_given_BG = sub_p_given_BG / devide
                else:
                    p_given_FG += sub_p_given_FG / devide
                    p_given_BG += sub_p_given_BG / devide
            # create value probability dictionary
            p_BG = p_given_BG/(p_given_BG+p_given_FG)
            p_FG = p_given_FG / (p_given_BG + p_given_FG)
            BG_Sval_p_dict = dict(zip(S_values, p_BG))
            FG_Sval_p_dict = dict(zip(S_values, p_FG))

        # calc probabilty maps
        BG_p_map = np.vectorize(BG_Sval_p_dict.get)(crop_frame_S)
        FG_p_map = np.vectorize(FG_Sval_p_dict.get)(crop_frame_S)
        if show_pmap:
            fig = plt.figure()
            a = fig.add_subplot(1, 2, 1)
            imgplot = plt.imshow(FG_p_map)
            a.set_title('FG_p_map')
            a = fig.add_subplot(1, 2, 2)
            imgplot = plt.imshow(BG_p_map)
            a.set_title('BG_p_map')
            plt.show()

        # calc alpha based on Geodesic distance weighted with likelihood
        r = R
        w_BG = BG_p_map / (dist_map_BG ** r)
        w_BG[dist_map_BG == 0] = 0
        w_FG = FG_p_map / (dist_map_FG ** r)
        w_FG[dist_map_FG == 0] = 0
        alpha = np.zeros_like(frame_S, np.float64)
        alpha[y:y+hight, x:x+width] = w_FG/(w_FG + w_BG)
        alpha[alpha < bot_alpha] = 0
        alpha[alpha > top_alpha] = 1

        alpha = np.uint8(alpha*255)

        # show W
        if show_w:
            fig = plt.figure()
            a = fig.add_subplot(1, 2, 1)
            imgplot = plt.imshow(w_FG)
            a.set_title('w_FG')
            a = fig.add_subplot(1, 2, 2)
            imgplot = plt.imshow(w_BG)
            a.set_title('w_BG')
            plt.show()

        # show alpha
        if show_alpha:
            fig, ax = plt.subplots(1)
            ax.imshow(alpha[y:y+hight, x:x+width])
            plt.show()

        # show frame
        out.write(alpha)
    # Release video
    cap.release()
    out.release()

    # Close windows
    cv2.destroyAllWindows()
    pbar.close()


def unstable(shaky, stable, alpha, shaky_alpha):
    # Read input video
    cap = cv2.VideoCapture(shaky)
    cap2 = cv2.VideoCapture(stable)
    cap3 = cv2.VideoCapture(alpha)

    # Set up output video
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(shaky_alpha, fourcc, fps, (w, h), isColor=False)
    pbar = tqdm(total=(n_frames - 1), desc='unstable', position=0, leave=True)

    for i in range(n_frames - 1):
        pbar.update(1)
        success, curr = cap.read()
        success2, prev = cap2.read()
        success3, alpha = cap3.read()
        if not success or not success2 or not success3:
            break
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        alpha = cv2.cvtColor(alpha, cv2.COLOR_BGR2GRAY)
        alpha[alpha < 10] = 0

        # Detect feature points in original frame
        prev_pts = cv2.goodFeaturesToTrack(prev_gray,
                                           maxCorners=1000,
                                           qualityLevel=0.01,
                                           minDistance=1,
                                           blockSize=3)

        # Calculate optical flow (i.e. track feature points)
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

        # Filter only valid points, with lowest error
        err = err[status == 1]
        min_errors_ind = np.argpartition(err, int(len(err) * 9 / 10), axis=0)[:int(len(err) * 9 / 10)]
        prev_pts = prev_pts[status == 1][min_errors_ind]
        curr_pts = curr_pts[status == 1][min_errors_ind]

        # Find transformation matrix
        m = cv2.estimateRigidTransform(prev_pts, curr_pts, fullAffine=True)  # will only work with OpenCV-3 or less

        # Apply affine wrapping to the given frame
        frame_unstabilized = np.uint8(cv2.warpAffine(alpha, m, (w, h)))

        # Write the frame to the file
        out.write(frame_unstabilized)

    # Release video
    cap.release()
    out.release()
    # Close windows
    cv2.destroyAllWindows()
    pbar.close()

def put_in_background(stab_vid, alpha, back_im, new_back):
    # Read input video
    cap1 = cv2.VideoCapture(alpha)
    cap2 = cv2.VideoCapture(stab_vid)

    # Get frame count
    n_frames = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))

    # Set up output video
    w = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap2.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(new_back, fourcc, fps, (w, h))

    back = np.array(cv2.imread(back_im), dtype='float64')
    back = cv2.resize(back, (w, h))
    pbar = tqdm(total=(n_frames - 1), desc='put_in_background', position=0, leave=True)

    for i in range(n_frames):
        pbar.update(1)
        # Read next frame
        success1, alpha = cap1.read()
        success2, frame2 = cap2.read()
        if not success1 or not success2:
            break
        alpha = np.array(cv2.cvtColor(alpha, cv2.COLOR_BGR2GRAY), dtype='float64') / 255
        alpha[alpha < 0.05] = 0
        alpha_new = np.zeros_like(frame2, dtype='float64')
        for j in range(3):
            alpha_new[:, :, j] = alpha

        frame2 = np.array(frame2, dtype='float64')
        frame_out = alpha_new * frame2 + (1 - alpha_new) * back
        frame_out = np.uint8(frame_out)

        out.write(frame_out)

    # Release video
    cap1.release()
    cap2.release()
    out.release()
    pbar.close()
    # Close windows
    cv2.destroyAllWindows()
