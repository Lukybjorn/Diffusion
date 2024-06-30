import os
import cv2
import numpy as np
import pandas as pd
from openpyxl import load_workbook
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def get_data():
    # a = float(input("Distance between glass rod to cell (cm): "))
    # b = float(input("Distance between cell to board (cm): "))
    # c = float(input("Width of the cell (cm): "))
    # return a, b, c
    pass

def fit_function(x_data, y_data, hight_ratio, width_ratio):

    plt.scatter(x_data, y_data)
    plt.show()
    
    x_min = int(input("Enter the first x value: "))
    x_max = int(input("enter the last x value: "))
    

    n_x_data = np.concatenate((x_data[:x_min+1], x_data[x_max:])) * width_ratio
    n_y_data = np.concatenate((y_data[:x_min+1], y_data[x_max:])) * hight_ratio

    # Define the model function to fit
    def func(x, A, B):
        return A * x + B

    # Fit the model to the data
    popt, pcov = curve_fit(func, n_x_data, n_y_data, p0=[1, 1])
    A, B = popt

    # Calculate the fitted values
    y_fit = func(n_x_data, *popt)

    # Calculate R^2 value
    residuals = n_y_data - y_fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((n_y_data - np.mean(n_y_data))**2)
    r_squared = 1 - (ss_res / ss_tot)

    # Extract standard deviations of the parameters from the covariance matrix
    perr = np.sqrt(np.diag(pcov))
    
    return {"x_fit": n_x_data, "y_fit": y_fit, "slope": A, "intersect": B, "slope_e": perr[0], "intersect_e": perr[1], "r_squared": r_squared}

def fit_plot(x_data, y_data, fit):
    
    x_fit = fit["x_fit"]
    y_fit = fit["y_fit"]
    slope = fit["slope"]
    slope_e = fit["slope_e"]
    intersect = fit["intersect"]
    intersect_e = fit["intersect_e"]
    r_squared = fit["r_squared"]
    
    print(f"Function: y = A*x + B")
    print(f"Fitted values: slope (A) = {slope:.6f} ± {slope_e:.6f}, intersect (B) = {intersect:.6f} ± {intersect_e:.6f}")
    print(f"R^2 value: {r_squared:.6f}")

    plt.scatter(x_data, y_data, label='Data')
    plt.plot(x_fit, y_fit, label='Fit', color='red')
    plt.xlim(0, max(x_data))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid()
    plt.show()

def image_data(image_file):
    image = cv2.imread(image_file)
    # image = cv2.flip(cv2.imread('pic8_crop.jpg'), 1)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    hight_pix, width_pix = image.shape[0:2]

    # Define a brightness threshold
    # Adjust this value based on your image; a value around 200 can work for very bright pixels
    threshold_value = 190
    _, bright_mask = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
    
    
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Brightness Mask')
    plt.imshow(bright_mask, cmap='gray')
    # plt.axis('off')
    
    plt.show()
    
    x_data = []
    y_data = []
    
    Y = list(range(len(bright_mask)))
    

    for y in Y:
        for x in range(len(bright_mask[y])):
            if bright_mask[y][x] != 0:
                x_data.append(x)
                y_data.append(y)
    
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    
    X = []
    Y = []
    
    x_set = set(x_data)
    for x_val in x_set:
        X.append(x_val)
        indexes = np.where(x_data == x_val)[0]
        Y.append(np.average(y_data[indexes]))
            
    return np.array(X), np.array(Y), width_pix, hight_pix
    # return 1, 2

def adjust_data(x_data, y_data, fit, param):
    a, b, c = param
    
    m = fit["slope"]
    n = fit["intersect"]
    
    new_x_data = (m*x_data + n)*a / (a + b)
    new_y_data = abs(m*x_data + n - y_data) / (c*b)
    
    plt.plot(new_x_data, new_y_data)
    plt.show()
    
    return new_x_data, new_y_data

def get_jpg_files(folder_path, particle):
    jpg_files = []
    for file in os.listdir(folder_path):
        if particle in file and "crop" in file:
            jpg_files.append(os.path.join(folder_path, file))
    if not jpg_files:
        raise Exception("No images with this name")
    return jpg_files


def main(particles: list):
    height_cm = 25
    width_cm = 25 
    # parameters = get_data()
    parameters = (17, 111, 2)
    for particle in particles:
        excel_path = f"results_{particle}.xlsx"
        folder_path = '.'
        jpg_files = get_jpg_files(folder_path, particle)

        # Check if the Excel file already exists
        if os.path.exists(excel_path):
            # Load the existing workbook
            book = load_workbook(excel_path)
            # Create an ExcelWriter object with the existing workbook
            with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a') as writer:
                writer.book = book
                writer.sheets = {ws.title: ws for ws in book.worksheets}
                for image_file in jpg_files:
                    print(f"Processing image: {image_file}")
                    x_data, y_data, width_pix, height_pix = image_data(image_file)
                    
                    height_ratio = height_cm / height_pix
                    width_ratio = width_cm / width_pix
                    
                    fit = fit_function(x_data, y_data, height_ratio, width_ratio)
                    x_data = x_data * width_ratio
                    y_data = y_data * height_ratio
                    fit_plot(x_data, y_data, fit)
                    gauss_x, gauss_y = adjust_data(x_data, y_data, fit, parameters)
                    df = pd.DataFrame({"X": gauss_x, "Y": gauss_y})
                    df.to_excel(writer, index=False, sheet_name=image_file[2:-4])
        else:
            # Create a new Excel file and add the sheets
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                for image_file in jpg_files:
                    print(f"Processing image: {image_file}")
                    x_data, y_data, width_pix, height_pix = image_data(image_file)

                    height_ratio = height_cm / height_pix
                    width_ratio = width_cm / width_pix
                    
                    fit = fit_function(x_data, y_data, height_ratio, width_ratio)
                    x_data = x_data * width_ratio
                    y_data = y_data * height_ratio
                    fit_plot(x_data, y_data, fit)
                    gauss_x, gauss_y = adjust_data(x_data, y_data, fit, parameters)
                    df = pd.DataFrame({"X": gauss_x, "Y": gauss_y})
                    df.to_excel(writer, index=False, sheet_name=image_file[2:-4])
        

if __name__ == "__main__":
    particles = ["Ag", "NH4", "KNO3", "KCl"]
    main(particles)
