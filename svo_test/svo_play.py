import sys
import os

import cv2
import pyzed.sl as sl

# Create a ZED camera object
zed = sl.Camera()
exit_signal = False

# Set SVO path for playback
cwd = os.getcwd()
svo_name = "j-turn_obstacle_avoidance.svo"
input_path = os.path.join(cwd, "svo_test", svo_name)
init_parameters = sl.InitParameters()
init_parameters.set_from_svo_file(input_path)

# Open the ZED
zed = sl.Camera()
err = zed.open(init_parameters)

svo_image = sl.Mat()

while not exit_signal:
  if zed.grab() == sl.ERROR_CODE.SUCCESS:
    # Read side by side frames stored in the SVO
    # zed.retrieve_image(svo_image, sl.VIEW.SIDE_BY_SIDE)
    zed.retrieve_image(svo_image, sl.VIEW.LEFT)
    # Get frame count
    svo_position = zed.get_svo_position()
    image_ocv = svo_image.get_data()
    # show the image
    cv2.imshow("Image", image_ocv)
    cv2.waitKey(1)

  elif zed.grab() == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
    exit_signal = True
    print("SVO end has been reached. Looping back to first frame")
    zed.set_svo_position(0)
