#
#  Assignment 3
#
#  Group 09:
#  <Sarthak Bharatbhai Gandhi> <sbgandhi@mun.ca>
#  <Foramben Hasmukhkumar Bhalsod> <fhbhalsod@mun.ca>

####################################################################################
# Imports
####################################################################################

import flowpy
import matplotlib.pyplot as plt
import cv2
import numpy as np

#####################################################################################
#  Implementation - helper functions and question wise result generators
#####################################################################################

def visualize_ground_truth(flow_file, img_1_path, img_2_path):
    flow = flowpy.flow_read(flow_file)
    img = flowpy.flow_to_rgb(flow)

    img1 = cv2.imread(img_1_path)
    img2 = cv2.imread(img_2_path)

    img1 = cv2.resize(img1, (img.shape[1], img.shape[0]))
    img2 = cv2.resize(img2, (img.shape[1], img.shape[0]))

    vis = np.concatenate((img1, img2, img), axis=1)

    cv2.imshow('Optical Flow Visualization', vis)
    cv2.imwrite('output_question1.png', vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def compute_and_visualize_optical_flow(image_path_initial, image_path_final, ground_truth_flow):
    initial_frame = cv2.imread(image_path_initial)
    final_frame = cv2.imread(image_path_final)

    ground_truth_flow = flowpy.flow_read(ground_truth_flow)

    b_initial, g_initial, r_initial = cv2.split(initial_frame)
    b_final, g_final, r_final = cv2.split(final_frame)

   # Compute optical flow using Farneback's algorithm for each channel

    flow_b = cv2.calcOpticalFlowFarneback(b_initial, b_final, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flow_g = cv2.calcOpticalFlowFarneback(g_initial, g_final, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flow_r = cv2.calcOpticalFlowFarneback(r_initial, r_final, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Compute optical flow using Farneback's algorithm for green channel
    prefered_weights = [0.114, 0.587, 0.299]

    green_channel = prefered_weights[0] * b_initial + prefered_weights[1] * g_initial + prefered_weights[2] * r_initial
    green_channel_next = prefered_weights[0] * b_final + prefered_weights[1] * g_final + prefered_weights[2] * r_final

    flow_green = cv2.calcOpticalFlowFarneback(green_channel, green_channel_next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    # # Compute optical flow using Farneback's algorithm for all cases

    flow_array = {'flow_b': flow_b, 'flow_g': flow_g, 'flow_r': flow_r, 'flow_green': flow_green}
    
    # Calculate EPE for each flow
    epe_results = {}
    for key, flow in flow_array.items():
        epe = np.sqrt(np.sum((flow - ground_truth_flow) ** 2, axis=2))
        average_epe = np.nanmean(epe)
        epe_results[key] = average_epe
        print(f'Average EPE for {key}: {average_epe}')

        # Visualize the flow map (optional)
        hsv = np.zeros_like(initial_frame)
        hsv[..., 1] = 255

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # flow_img = flowpy.flow_to_rgb(flow, convert_to_bgr=False)
        cv2.imshow(f'Flow - {key}', rgb)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def compute_farneback_optical_flow(prev_img, next_img):
    prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow

def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow_map = -flow
    flow_map[:,:,0] += np.arange(w)
    flow_map[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow_map, None, cv2.INTER_LINEAR)
    return res

def estimate_middle_frame(initial_frame_path, final_frame_path, ground_truth_middle_frame_path):
    initial_frame = cv2.imread(initial_frame_path)
    final_frame = cv2.imread(final_frame_path)
    ground_truth_middle_frame = cv2.imread(ground_truth_middle_frame_path)

    # Compute forward and backward optical flow
    forward_flow = compute_farneback_optical_flow(initial_frame, final_frame)
    backward_flow = compute_farneback_optical_flow(final_frame, initial_frame)

    # Estimate middle frame using backward warping
    forward_middle_estimate = warp_flow(final_frame, forward_flow)
    backward_middle_estimate = warp_flow(initial_frame, backward_flow)

    # Calculate RMS error for both estimates
    forward_rms = np.sqrt(np.mean((forward_middle_estimate - ground_truth_middle_frame) ** 2))
    backward_rms = np.sqrt(np.mean((backward_middle_estimate - ground_truth_middle_frame) ** 2))

    # Optionally, save or display the estimated middle frames and print RMS errors
    cv2.imwrite('forward_middle_estimate.png', forward_middle_estimate)
    cv2.imwrite('backward_middle_estimate.png', backward_middle_estimate)

    print(f'Forward RMS error: {forward_rms}')
    print(f'Backward RMS error: {backward_rms}')

    return forward_middle_estimate, backward_middle_estimate


def compute_optical_flow_lucas_kanade(initial_frame_path, final_frame_path, flow_map_path):
    # Load initial and final frames
    initial_frame = cv2.imread(initial_frame_path)
    final_frame = cv2.imread(final_frame_path)

    # Convert frames to grayscale
    gray1 = cv2.cvtColor(initial_frame, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(final_frame, cv2.COLOR_BGR2GRAY)

    # Detect keypoints to track using Shi-Tomasi corner detection
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    p0 = cv2.goodFeaturesToTrack(gray1, mask=None, **feature_params)

    # Calculate optical flow using Lucas-Kanade method
    p1, st, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None)

    # Create a mask to discard outliers
    mask = np.zeros_like(gray1)
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # Draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel().astype(int)
        c, d = old.ravel().astype(int)
        mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)
        final_frame = cv2.circle(final_frame, (a, b), 5, (0, 255, 0), -1)

    # Calculate flow map from the tracked keypoints
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Save the estimated optical flow map
    # cv2.imwrite(flow_map_path, flow)

    flowpy.flow_write(flow_map_path, flow)

def computeOpticalFlow(initial_frame_path, final_frame_path, flow_map_path, model_dir_path=None):
    # Read the initial and final frames
    initial_frame = cv2.imread(initial_frame_path)
    final_frame = cv2.imread(final_frame_path)
    
    # Convert frames to grayscale
    initial_gray = cv2.cvtColor(initial_frame, cv2.COLOR_BGR2GRAY)
    final_gray = cv2.cvtColor(final_frame, cv2.COLOR_BGR2GRAY)
    
    # Compute the optical flow using Farnebäck's algorithm
    flow = cv2.calcOpticalFlowFarneback(initial_gray, final_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    # Save the computed flow map
    flowpy.flow_write(flow_map_path, flow)

    # Optionally, visualize the computed flow map
    flow_img = flowpy.flow_to_rgb(flow)
    
    cv2.imshow('Optical Flow', flow_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def Q1_results():
    print('Generating results for Q1...')
    # Visualize the ground truth flow
    visualize_ground_truth('data/Grove3/flow10.flo', 'data/Grove3/frame10.png', 'data/Grove3/frame11.png')
    visualize_ground_truth('data/RubberWhale/flow10.flo', 'data/RubberWhale/frame10.png', 'data/RubberWhale/frame11.png')

def Q2_results():
    print('Generating results for Q2...')
    compute_and_visualize_optical_flow('data/Grove3/frame10.png', 'data/Grove3/frame11.png', 'data/Grove3/flow10.flo') 
    compute_and_visualize_optical_flow('data/RubberWhale/frame10.png', 'data/RubberWhale/frame11.png', 'data/RubberWhale/flow10.flo')
    
def Q3_results():
    print('Generating results for Q3...')
    compute_and_visualize_optical_flow('data/Grove3/frame11.png', 'data/Grove3/frame10.png', 'data/Grove3/flow10.flo')
    compute_and_visualize_optical_flow('data/RubberWhale/frame11.png', 'data/RubberWhale/frame10.png', 'data/RubberWhale/flow10.flo')

def Q4_results():
    print('Generating results for Q4...')
    # Estimate the middle frame using the computed optical flow

    # estimate_middle_frame('data/Grove3/frame10.png', 'data/Grove3/frame11.png', 'data/Grove3/frame10.png')
    # estimate_middle_frame('data/RubberWhale/frame10.png', 'data/RubberWhale/frame11.png', 'data/RubberWhale/frame10.png')
    

def Q5_results():
    print('Generating results for Q5...')
    # Estimate the middle frame using the computed optical flow
    # computeOpticalFlow('data/Grove3/frame10.png', 'data/Grove3/frame11.png', 'grove_flow10.flo')
    # computeOpticalFlow('data/RubberWhale/frame10.png', 'data/RubberWhale/frame11.png', 'rubberwhale_flow10.flo')
    # compute_optical_flow_lucas_kanade('data/Grove3/frame10.png', 'data/Grove3/frame11.png', 'grove_flow10.flo')
    # compute_optical_flow_lucas_kanade('data/RubberWhale/frame10.png', 'data/RubberWhale/frame11.png', 'rubberwhale_flow10.flo')

#########################################################################################
# Calls to generate the results
#########################################################################################
if __name__ == "__main__":
    Q1_results()
    Q2_results()
    Q3_results()
    Q4_results()
    Q5_results()



# Average EPE for RGB: 1.339880347251892
# Average EPE for Green Channel: 1.3682315349578857
# Average EPE for Grayscale: 1.339880347251892
# Average EPE for RGB: 7.498711585998535
# Average EPE for Green Channel: 7.493842124938965
# Average EPE for Grayscale: 7.498711585998535