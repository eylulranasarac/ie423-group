import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import wiener

image_path = "ProjectPart3\\rawimage\\0012.jpg"
# image_path = "C:\\Users\\EylülRanaSaraç\\OneDrive - boun.edu.tr\\Masaüstü\\IE 423\\Project Part 2\\golden-girlz\\ProjectPart3\\rawimage\\0012.jpg"

original_image = cv2.imread(image_path)

# Convert the colored image from BGR to RGB for correct color display with Matplotlib
colored_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

grayscale_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# enhance image quality with reducing noice
blurred_image = cv2.GaussianBlur(grayscale_image, (5, 5), 0)

# sharpening
sharpening_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
sharpened_image = cv2.filter2D(blurred_image, -1, sharpening_kernel)

# enhance contrast with histogram equalization
equalized_image = cv2.equalizeHist(sharpened_image)

# Show results
plt.figure(figsize=(12, 12))
plt.subplot(2, 5, 1)
plt.imshow(colored_image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 5, 2)
plt.imshow(grayscale_image, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')

plt.subplot(2, 5, 3)
plt.imshow(blurred_image, cmap='gray')
plt.title('Blurred Image')
plt.axis('off')

plt.subplot(2, 5, 4)
plt.imshow(sharpened_image, cmap='gray')
plt.title('Sharpened Image')
plt.axis('off')

plt.subplot(2, 5, 5)
plt.imshow(equalized_image, cmap='gray')
plt.title('Equalized Image')
plt.axis('off')

#plt.show()


# MORPHOLOGICAL EROSION

# Define a vertical line structuring element
kernel = np.array([[0, 1, 0], 
                   [0, 1, 0], 
                   [0, 1, 0]], dtype=np.uint8)

eroded_image = cv2.erode(equalized_image, kernel)

# Thresholding
threshold_value = 0.5 * 255  # Adjusting the threshold value to the scale of [0, 255]
_, binary_image = cv2.threshold(eroded_image, threshold_value, 255, cv2.THRESH_BINARY)

# Labeling
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, 8, cv2.CV_32S)

# Ignoring small objects (less than 4 pixels)
min_pixel_size = 4
for label in range(1, num_labels):
    if stats[label, cv2.CC_STAT_AREA] < min_pixel_size:
        binary_image[labels == label] = 0

plt.subplot(2, 5, 6)
plt.imshow(binary_image, cmap='gray')
plt.title("Segmented Image")
plt.axis('off')
#plt.show()


# Apply Wiener Filter
filtered_image = wiener(binary_image, (2, 2))

plt.subplot(2, 5, 7)
plt.imshow(filtered_image, cmap='gray')
plt.title("Filtered Image")
plt.axis('off')

# Find contours (ROIs) in the filtered image
contours, _ = cv2.findContours(filtered_image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Identify the largest contour (ROI)
largest_contour = max(contours, key=cv2.contourArea)

# Create an empty image to draw the largest ROI
output_image = np.zeros_like(binary_image)
cv2.drawContours(output_image, [largest_contour], -1, (255, 255, 255), thickness=cv2.FILLED)

# Display the results using matplotlib

plt.subplot(2, 5, 10)
plt.imshow(output_image, cmap='gray')
plt.title("Detected Fabric Defect")
plt.axis('off')

plt.subplots_adjust(hspace=0)
plt.show()