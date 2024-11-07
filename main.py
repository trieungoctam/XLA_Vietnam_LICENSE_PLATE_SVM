import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox
import math

# Khởi tạo cửa sổ Tkinter
root = tk.Tk()
root.title("Image Processing Slides")
root.geometry("800x650")

# Khai báo các hằng số
GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)
ADAPTIVE_THRESH_BLOCK_SIZE = 19
ADAPTIVE_THRESH_WEIGHT = 9

Min_char = 0.01
Max_char = 0.09

# Đường dẫn đến mô hình SVM đã huấn luyện
SVM_MODEL_PATH = "svm.xml"
RESIZED_IMAGE_WIDTH = 30
RESIZED_IMAGE_HEIGHT = 60

# Tải mô hình SVM
svm_model = cv2.ml.SVM_load(SVM_MODEL_PATH)

# Danh sách trạng thái, tiêu đề và chỉ số hiện tại
image_states = []
state_titles = []
current_state_index = 0

# Hàm hiển thị ảnh lên Label Tkinter với kích thước điều chỉnh
def display_image(img, label):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_pil.thumbnail((800, 500), Image.LANCZOS)  # Giới hạn kích thước ảnh
    img_tk = ImageTk.PhotoImage(img_pil)
    label.config(image=img_tk)
    label.image = img_tk  # Lưu trữ ảnh để tránh bị xóa khi kết thúc hàm

# Hàm cập nhật tiêu đề
def update_title():
    label_title.config(text=state_titles[current_state_index])


# Hàm xử lý ảnh và lưu các bước vào danh sách
def process_image(image_path):
    n = 1
    global image_states, current_state_index
    image_states.clear()  # Xóa các trạng thái cũ
    state_titles.clear()  # Xóa các tiêu đề cũ
    current_state_index = 0  # Đặt lại chỉ số trạng thái

    # Đọc và hiển thị ảnh gốc
    imgOriginal = cv2.imread(image_path)
    imgOriginal = cv2.resize(imgOriginal, dsize=(1920, 1080))
    if imgOriginal is None:
        messagebox.showerror("Error", "Cannot open image file.")
        return
    image_states.append(imgOriginal.copy())
    state_titles.append("Original Image")

    # Chuyển đổi ảnh sang ảnh xám
    imgGrayscale = extractValue(imgOriginal)
    image_states.append(cv2.cvtColor(imgGrayscale, cv2.COLOR_GRAY2BGR))  # Chuyển lại về BGR để hiển thị
    state_titles.append("Grayscale Image")

    # Tăng độ tương phản
    imgMaxContrastGrayscale = maximizeContrast(imgGrayscale)
    image_states.append(cv2.cvtColor(imgMaxContrastGrayscale, cv2.COLOR_GRAY2BGR))
    state_titles.append("Maximized Contrast")

    # Làm mịn bằng Gaussian Blur
    height, width = imgGrayscale.shape
    imgBlurred = np.zeros((height, width, 1), np.uint8)
    imgBlurred = cv2.GaussianBlur(imgMaxContrastGrayscale, GAUSSIAN_SMOOTH_FILTER_SIZE, 0)
    image_states.append(cv2.cvtColor(imgBlurred, cv2.COLOR_GRAY2BGR))
    state_titles.append("Gaussian Blurred")

    # Nhị phân hóa bằng ngưỡng thích ứng
    imgThreshplate = cv2.adaptiveThreshold(imgBlurred, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT)
    image_states.append(cv2.cvtColor(imgThreshplate, cv2.COLOR_GRAY2BGR))
    state_titles.append("Adaptive Threshold")

    imgGrayscaleplate = imgGrayscale.copy()
    # Phát hiện cạnh bằng Canny Edge
    canny_image = cv2.Canny(imgThreshplate, 250, 255)
    image_states.append(cv2.cvtColor(canny_image, cv2.COLOR_GRAY2BGR))
    state_titles.append("Canny Edge Detection")

    kernel = np.ones((3, 3), np.uint8)
    dilated_image = cv2.dilate(canny_image, kernel, iterations=1)  # Dilation
    image_states.append(cv2.cvtColor(dilated_image, cv2.COLOR_GRAY2BGR))
    state_titles.append("Dilated Image")

    # Lọc biển số bằng contour
    imgContour = imgOriginal.copy()
    ###### Draw contour and filter out the license plate  #############
    contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]  # Lấy 10 contours có diện tích lớn nhất

    screenCnt = []
    for c in contours:
        peri = cv2.arcLength(c, True)  # Tính chu vi
        approx = cv2.approxPolyDP(c, 0.06 * peri, True)  # làm xấp xỉ đa giác, chỉ giữ contour có 4 cạnh
        [x, y, w, h] = cv2.boundingRect(approx.copy())
        if (len(approx) == 4):
            screenCnt.append(approx)
            cv2.putText(imgContour, str(len(approx.copy())), (x, y), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 3)


    if screenCnt is None:
        detected = 0
        print("No plate detected")
    else:
        detected = 1

    if detected == 1:
        copyImage = imgContour.copy()
        for screen in screenCnt:
            cv2.drawContours(copyImage, [screen], -1, (0, 255, 0), 3)  # Khoanh vùng biển số xe
        image_states.append(copyImage)
        state_titles.append("Detected Plate")

        for screen in screenCnt:
            cv2.drawContours(imgContour, [screen], -1, (0, 255, 0), 3)  # Khoanh vùng biển số xe
            ############## Find the angle of the license plate #####################
            (x1, y1) = screen[0, 0]
            (x2, y2) = screen[1, 0]
            (x3, y3) = screen[2, 0]
            (x4, y4) = screen[3, 0]
            array = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            sorted_array = array.sort(reverse=True, key=lambda x: x[1])
            (x1, y1) = array[0]
            (x2, y2) = array[1]
            doi = abs(y1 - y2)
            ke = abs(x1 - x2)
            angle = math.atan(doi / ke) * (180.0 / math.pi)

            ####################################

            ########## Crop out the license plate and align it to the right angle ################

            mask = np.zeros(imgGrayscaleplate.shape, np.uint8)
            new_image = cv2.drawContours(mask, [screen], 0, 255, -1, )

            # Cropping
            (x, y) = np.where(mask == 255)
            (topx, topy) = (np.min(x), np.min(y))
            (bottomx, bottomy) = (np.max(x), np.max(y))

            roi = imgContour[topx:bottomx, topy:bottomy]
            imgThresh = imgThreshplate[topx:bottomx, topy:bottomy]
            ptPlateCenter = (bottomx - topx) / 2, (bottomy - topy) / 2

            if x1 < x2:
                rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, -angle, 1.0)
            else:
                rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, angle, 1.0)

            roi = cv2.warpAffine(roi, rotationMatrix, (bottomy - topy, bottomx - topx))
            imgThresh = cv2.warpAffine(imgThresh, rotationMatrix, (bottomy - topy, bottomx - topx))
            roi = cv2.resize(roi, (0, 0), fx=3, fy=3)
            imgThresh = cv2.resize(imgThresh, (0, 0), fx=3, fy=3)

            # cv2.imshow("imgThresh",imgThresh)
            # cv2.imshow("roi",roi)

            ##################################

            #################### Prepocessing and Character segmentation ####################
            kerel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            thre_mor = cv2.morphologyEx(imgThresh, cv2.MORPH_DILATE, kerel3)
            cont, hier = cv2.findContours(thre_mor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            cv2.drawContours(roi, cont, -1, (100, 255, 255), 2)

            ##################### Filter out characters #################
            char_x_ind = {}
            char_x = []
            height, width, _ = roi.shape
            roiarea = height * width

            for ind, cnt in enumerate(cont):
                (x, y, w, h) = cv2.boundingRect(cont[ind])
                ratiochar = w / h
                char_area = w * h

                if (Min_char * roiarea < char_area < Max_char * roiarea) and (0.25 < ratiochar < 0.7):
                    if x in char_x:  # Sử dụng để dù cho trùng x vẫn vẽ được
                        x = x + 1
                    char_x.append(x)
                    char_x_ind[x] = ind

            ############ Character recognition ##########################

            char_x = sorted(char_x)
            strFinalString = ""
            first_line = ""
            second_line = ""

            for idx, i in enumerate(char_x):
                (x, y, w, h) = cv2.boundingRect(cont[char_x_ind[i]])
                cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

                imgROI = thre_mor[y:y + h, x:x + w]  # Crop the characters

                imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))  # resize image
                npaROIResized = imgROIResized.reshape(1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT).astype(np.float32)

                _, npaResults = svm_model.predict(npaROIResized)
                result = int(npaResults[0][0])
                if result <= 9:  # Neu la so thi hien thi luon
                    result = str(result)
                else:  # Neu la chu thi chuyen bang ASCII
                    result = chr(result)

                cv2.putText(roi, result, (x, y + 50), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 0), 3)

                # Phân loại ký tự theo hàng trên hoặc dưới
                if y < height / 3:
                    first_line += result
                else:
                    second_line += result

            roi = cv2.resize(roi, None, fx=0.75, fy=0.75)
            image_states.append(roi)
            state_titles.append("Detected Characters")

            cv2.putText(imgOriginal, first_line + "-" + second_line ,(topy ,topx),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 2)
            image_states.append(imgOriginal)
            state_titles.append("License Plate")
            n = n + 1


    # Hiển thị trạng thái đầu tiên và cập nhật tiêu đề
    display_image(image_states[current_state_index], label_img)
    update_title()

    # Hiển thị các nút "Next" và "Back"
    btn_back.pack(side="left", padx=10, pady=10)
    btn_next.pack(side="right", padx=10, pady=10)

# Hàm chuyển sang trạng thái tiếp theo
def next_state():
    global current_state_index
    if current_state_index < len(image_states) - 1:
        current_state_index += 1
        display_image(image_states[current_state_index], label_img)
        update_title()  # Cập nhật tiêu đề

# Hàm quay lại trạng thái trước
def prev_state():
    global current_state_index
    if current_state_index > 0:
        current_state_index -= 1
        display_image(image_states[current_state_index], label_img)
        update_title()  # Cập nhật tiêu đề

# Hàm chuyển đổi sang hệ màu HSV và trích xuất kênh 'Value'
def extractValue(imgOriginal):
    height, width, numChannels = imgOriginal.shape
    imgHSV = np.zeros((height, width, 3), np.uint8)
    imgHSV = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2HSV)
    imgHue, imgSaturation, imgValue = cv2.split(imgHSV)
    return imgValue

# Hàm tăng cường độ tương phản
def maximizeContrast(imgGrayscale):
    height, width = imgGrayscale.shape
    imgTopHat = np.zeros((height, width, 1), np.uint8)
    imgBlackHat = np.zeros((height, width, 1), np.uint8)
    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)) #tạo bộ
    imgTopHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_TOPHAT, structuringElement, iterations = 10)
    imgBlackHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_BLACKHAT, structuringElement, iterations = 10)
    imgGrayscalePlusTopHat = cv2.add(imgGrayscale, imgTopHat)
    imgGrayscalePlusTopHatMinusBlackHat = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)
    return imgGrayscalePlusTopHatMinusBlackHat

# Hàm mở hộp thoại chọn ảnh và gọi xử lý ảnh
def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        process_image(file_path)

# Label để hiển thị tiêu đề
label_title = tk.Label(root, text="", font=("Helvetica", 16))
label_title.pack(pady=10)

# Nút mở ảnh
btn_open = tk.Button(root, text="Open Image", command=open_file)
btn_open.pack(pady=10)

# Nút chuyển sang trạng thái tiếp theo và quay lại (ẩn khi khởi tạo)
btn_next = tk.Button(root, text="Next", command=next_state)
btn_back = tk.Button(root, text="Back", command=prev_state)

# Label để hiển thị ảnh đã chọn
label_img = tk.Label(root)
label_img.pack()

# Bắt đầu vòng lặp của Tkinter
root.mainloop()
