import cv2
import numpy as np

# Load the image
image = cv2.imread('/mnt/logicNAS/Exchange/vivek/Industrial_100/27.Messrohr/with_hand/IMG_20190206_122608.jpg')

# Define the region of interest (ROI) coordinates for the object
roi = [(0, 0), (1000, 1000)]  # Replace with actual coordinates

# Create a blank mask
mask = np.zeros_like(image[:, :, 0])

# Draw a white polygon corresponding to the object on the mask
cv2.fillPoly(mask, [np.array(roi)], 255)

# Save the mask
cv2.imwrite('segmentation_mask.jpg', mask)

# display the mask
#cv2.imshow('mask', mask)

# save the masked image
masked_image = cv2.bitwise_and(image, image, mask=mask)
cv2.imwrite('masked_image.jpg', masked_image)
