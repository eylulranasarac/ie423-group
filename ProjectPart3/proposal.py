import cv2
import matplotlib.pyplot as plt
import numpy as np

image_path = r"C:\project part 1\0015.jpg"

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
plt.subplot(1, 5, 1)
plt.imshow(colored_image)
plt.title('Original Image')

plt.subplot(1, 5, 2)
plt.imshow(grayscale_image, cmap='gray')
plt.title('Grayscale Image')

plt.subplot(1, 5, 3)
plt.imshow(blurred_image, cmap='gray')
plt.title('Blurred Image')

plt.subplot(1, 5, 4)
plt.imshow(sharpened_image, cmap='gray')
plt.title('Sharpened Image')

plt.subplot(1, 5, 5)
plt.imshow(equalized_image, cmap='gray')
plt.title('Equalized Image')

plt.show()
