import flowpy
import matplotlib.pyplot as plt
import cv2
import numpy as np

def visualize_ground_truth(flow_file, img_1_path, img_2_path):
    flow = flowpy.flow_read(flow_file)
    img = flowpy.flow_to_rgb(flow)

    img1 = cv2.imread(img_1_path)
    img2 = cv2.imread(img_2_path)

    # visualize the color coding of the optical flow
    img3 = flowpy.attach_calibration_pattern(img, flow_max_radius=flowpy.get_flow_max_radius(flow))



    # visualize the optical flow on the image
    img1 = cv2.resize(img1, (img.shape[1], img.shape[0]))
    img2 = cv2.resize(img2, (img.shape[1], img.shape[0]))
    img3 = cv2.resize(img, (img.shape[1], img.shape[0]))

    vis = np.concatenate((img1, img2, img, img3), axis=1)

    cv2.imshow('Optical Flow Visualization', vis)
    cv2.imwrite('output_question1.png', vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()






    # height, width, _ = flow.shape
    # image_ratio = height / width
    # max_radius = flowpy.get_flow_max_radius(flow)

    # return color

    # fig, (ax_1, ax_2) = plt.subplots(
    #     1, 2, gridspec_kw={"width_ratios": [1, image_ratio]}
    # )

    # ax_1.imshow(flowpy.flow_to_rgb(flow))
    # flowpy.attach_arrows(ax_1, flow)
    # flowpy.attach_coord(ax_1, flow)

    # flowpy.attach_calibration_pattern(ax_2, flow_max_radius=max_radius)

    # plt.show()


#     flow = flowpy.flow_read("tests/data/Dimetrodon.flo")
# height, width, _ = flow.shape

# image_ratio = height / width
# max_radius = flowpy.get_flow_max_radius(flow)

# fig, (ax_1, ax_2) = plt.subplots(
#     1, 2, gridspec_kw={"width_ratios": [1, image_ratio]}
# )

# ax_1.imshow(flowpy.flow_to_rgb(flow))
# flowpy.attach_arrows(ax_1, flow)
# flowpy.attach_coord(ax_1, flow)

# flowpy.attach_calibration_pattern(ax_2, flow_max_radius=max_radius)

# plt.show()



    # img1 = cv2.resize(img1, (img.shape[1], img.shape[0]))
    # img2 = cv2.resize(img2, (img.shape[1], img.shape[0]))
    # img3 = cv2.resize(img, (img.shape[1], img.shape[0]))

    # vis = np.concatenate((img1, img2, img, img3), axis=1)

    # cv2.imshow('Optical Flow Visualization', vis)
    # cv2.imwrite('output_question1.png', vis)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()




def compute_and_visualize_optical_flow(image_path_initial, image_path_final):
    initial_frame = cv2.imread(image_path_initial)
    final_frame = cv2.imread(image_path_final)
    gray_initial = cv2.cvtColor(initial_frame, cv2.COLOR_BGR2GRAY)
    gray_final = cv2.cvtColor(final_frame, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(gray_initial, gray_final, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    hsv = np.zeros_like(initial_frame)
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    cv2.imshow('Optical Flow', rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage

def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow_map = -flow
    flow_map[:,:,0] += np.arange(w)
    flow_map[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow_map, None, cv2.INTER_LINEAR)
    return res

def estimate_middle_frame(initial_frame_path, final_frame_path, forward_flow_path, backward_flow_path, ground_truth_middle_frame_path):
    initial_frame = cv2.imread(initial_frame_path)
    final_frame = cv2.imread(final_frame_path)
    forward_flow = flowpy.flow_read(forward_flow_path)
    backward_flow = flowpy.flow_read(backward_flow_path)
    ground_truth_middle_frame = cv2.imread(ground_truth_middle_frame_path)

    forward_middle_estimate = warp_flow(final_frame, forward_flow)
    backward_middle_estimate = warp_flow(initial_frame, backward_flow)

    forward_rms = np.sqrt(np.mean((forward_middle_estimate - ground_truth_middle_frame) ** 2))
    backward_rms = np.sqrt(np.mean((backward_middle_estimate - ground_truth_middle_frame) ** 2))

    cv2.imwrite('forward_middle_estimate.png', forward_middle_estimate)
    cv2.imwrite('backward_middle_estimate.png', backward_middle_estimate)

    print(f'Forward RMS error: {forward_rms}')
    print(f'Backward RMS error: {backward_rms}')

    return forward_middle_estimate, backward_middle_estimate

# Example usage
visualize_ground_truth('data/Grove3/flow10.flo', 'data/Grove3/frame10.png', 'data/Grove3/frame11.png')



# Visualize the flow
# flow_color = visualize_ground_truth('data/Grove3/flow10.flo', 'data/Grove3/frame10.png', 'data/Grove3/frame11.png')
# cv2.imshow("Flow Visualization", flow_color)
# cv2.waitKey(0)
# cv2.destroyAllWindows()





compute_and_visualize_optical_flow('data/Grove3/frame10.png', 'data/Grove3/frame11.png')
compute_and_visualize_optical_flow('data/Grove3/frame11.png', 'data/Grove3/frame10.png')

# estimate_middle_frame('path/to/frame10.png', 'path/to/frame11.png', 'path/to/forward_flow.flo', 'path/to/backward_flow.flo', 'path/to/frame10i11.png')

# import cv2
# import numpy as np
# import flowpy

def compute_farneback_optical_flow(prev_img, next_img):
    """
    Computes the Farneback optical flow between two images.
    """
    prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow

def warp_flow(img, flow):
    """
    Applies backward warping to an image using the given optical flow.
    """
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

# Example usage
estimate_middle_frame('data/Grove3/frame10.png', 'data/Grove3/frame11.png', 'data/Grove3/frame10i11.png')


# def compute_optical_flow_lucas_kanade(initial_frame_path, final_frame_path, flow_map_path):
#     # Load initial and final frames
#     initial_frame = cv2.imread(initial_frame_path)
#     final_frame = cv2.imread(final_frame_path)

#     # Convert frames to grayscale
#     gray1 = cv2.cvtColor(initial_frame, cv2.COLOR_BGR2GRAY)
#     gray2 = cv2.cvtColor(final_frame, cv2.COLOR_BGR2GRAY)

#     # Detect keypoints to track using Shi-Tomasi corner detection
#     feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
#     p0 = cv2.goodFeaturesToTrack(gray1, mask=None, **feature_params)

#     # Calculate optical flow using Lucas-Kanade method
#     p1, st, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None)

#     # Create a mask to discard outliers
#     mask = np.zeros_like(gray1)
#     good_new = p1[st == 1]
#     good_old = p0[st == 1]

#     # Draw the tracks
#     for i, (new, old) in enumerate(zip(good_new, good_old)):
#         a, b = new.ravel().astype(int)
#         c, d = old.ravel().astype(int)
#         mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)
#         final_frame = cv2.circle(final_frame, (a, b), 5, (0, 255, 0), -1)

#     # Calculate flow map from the tracked keypoints
#     flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

#     # Save the estimated optical flow map
#     # cv2.imwrite(flow_map_path, flow)

#     flowpy.flow_write(flow_map_path, flow)

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



# Example usage:
initial_frame_path = 'data/Grove3/frame10.png'
final_frame_path = 'data/Grove3/frame11.png'
flow_map_path = 'flow10.flo'
# compute_optical_flow_lucas_kanade(initial_frame_path, final_frame_path, flow_map_path)


computeOpticalFlow(initial_frame_path, final_frame_path, flow_map_path)