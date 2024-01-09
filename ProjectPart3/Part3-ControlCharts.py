import numpy as np
import matplotlib.pyplot as plt
import cv2

# Fotoğraf dosyasının adını ve yolunu belirtin
image_path = "C:/Users/ASUS/Desktop/423-part3/0012.jpg"

# OpenCV ile resmi yükle
# Grayscale olarak oku
grayscale_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Resmi Numpy dizisine dönüştür
original_data = np.array(grayscale_image)

# Function to calculate mean, standard deviation, and control limits for each row
def calculate_control_limits_per_row(data):
    num_rows, num_cols = data.shape
    row_means = np.mean(data, axis=1)
    row_std_devs = np.std(data, axis=1)

    # Calculate control limits for each row
    control_limits = {
        f"Row {i + 1}": {
            "Upper Limit": row_means[i] + 2 * row_std_devs[i],
            "Lower Limit": row_means[i] - 2 * row_std_devs[i]
        } for i in range(num_rows)
    }

    return control_limits

# Function to calculate mean, standard deviation, and control limits for each column
def calculate_control_limits_per_column(data):
    num_rows, num_cols = data.shape
    col_means = np.mean(data, axis=0)
    col_std_devs = np.std(data, axis=0)

    # Calculate control limits for each column
    control_limits = {
        f"Column {j + 1}": {
            "Upper Limit": col_means[j] + 2 * col_std_devs[j],
            "Lower Limit": col_means[j] - 2 * col_std_devs[j]
        } for j in range(num_cols)
    }

    return control_limits

# Function to find out-of-control points for each row or column
def find_out_of_control_points(data, control_limits):
    out_of_control_points = {}

    for i, (row_or_col, limits) in enumerate(control_limits.items()):
        upper_limit = limits['Upper Limit']
        lower_limit = limits['Lower Limit']

        # Find indices of points outside the control limits
        out_of_control_indices = np.where((data[i, :] > upper_limit) | (data[i, :] < lower_limit))[0]

        # Save the indices for each row or column
        out_of_control_points[row_or_col] = out_of_control_indices

    return out_of_control_points

# Function to set out-of-control points to black
def set_out_of_control_points_to_black(data, out_of_control_points):
    modified_data = np.copy(data)

    for row_or_col, indices in out_of_control_points.items():
        if "Row" in row_or_col:
            row_index = int(row_or_col.split()[1]) - 1
            modified_data[row_index, indices] = 0
        elif "Column" in row_or_col:
            col_index = int(row_or_col.split()[1]) - 1
            modified_data[indices, col_index] = 0

    return modified_data

# Call the function to get the modified data for rows
control_limits_per_row = calculate_control_limits_per_row(original_data)
out_of_control_points_per_row = find_out_of_control_points(original_data, control_limits_per_row)
modified_data_per_row = set_out_of_control_points_to_black(original_data, out_of_control_points_per_row)

# Call the function to get the modified data for columns
control_limits_per_column = calculate_control_limits_per_column(original_data)
out_of_control_points_per_column = find_out_of_control_points(original_data.T, control_limits_per_column)
modified_data_per_column = set_out_of_control_points_to_black(original_data.T, out_of_control_points_per_column)

# Orijinal ve değiştirilmiş veri setlerini plot et
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.imshow(original_data, cmap='gray', vmin=0, vmax=255)
plt.title("Orijinal Veri")

plt.subplot(1, 3, 2)
plt.imshow(modified_data_per_row, cmap='gray', vmin=0, vmax=255)
plt.title("Değiştirilmiş Veri (Row)")

plt.subplot(1, 3, 3)
plt.imshow(modified_data_per_column.T, cmap='gray', vmin=0, vmax=255)
plt.title("Değiştirilmiş Veri (Column)")

plt.show()
