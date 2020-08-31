import cv2, time, pandas
from datetime import datetime

# the first frame captured (background)
first_frame = None

status_list = [0, 0]
times = []
df = pandas.DataFrame(columns = ["Start", "End"])

# trigger the camera
video = cv2.VideoCapture(0)

while True:
    # check: if the video is running
    # frame: Numpy array, representing the next image captured
    check, current_frame = video.read()

    # no motion
    status = 0

    # print(check)
    # print(current_frame)

    # convert the first image to gray and make it blurry
    gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # initialize the first frame as background
    if first_frame is None:
        first_frame = gray
        # skip the following code and enter the next loop directly
        continue

    # calculate the difference of the current frame and the background
    difference = cv2.absdiff(first_frame, gray)

    # if the difference of the current frame and the background is > 30, make it white
    threshold_img = cv2.threshold(difference, 30, 255, cv2.THRESH_BINARY)[1]

    # remove black spots in big white area
    threshold_img = cv2.dilate(threshold_img, None, iterations = 3)

    # detect contours
    (cnts, _) = cv2.findContours(threshold_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # only count area >= 10000
    for contour in cnts:
        if cv2.contourArea(contour) < 10000:
            continue
        # else, draw the outline
        status = 1

        (x, y, w, h) = cv2.boundingRect(contour)
        # draw the rectangle in the current frame
        cv2.rectangle(current_frame, (x, y), (x+w, y+h), (0,255,0), 3)

    status_list.append(status)

    # status_list = status_list[-2:]

    # record the time when statue change occurs
    # if status_list[-1] == 1 and status_list[-2] == 0:
    #     times.append(datetime.now())
    # if status_list[-1] == 0 and status_list[-2] == 1:
    #     times.append(datetime.now())
    if status_list[-1] != status_list[-2]:
        times.append(datetime.now())

    cv2.imshow("gray", gray)
    cv2.imshow("difference", difference)
    cv2.imshow("threshold", threshold_img)
    cv2.imshow("frame", current_frame)

    # show the first frame of the video in gray
    cv2.imshow("Capturing", gray)

    # wait for 1 sec
    key = cv2.waitKey(1)

    # if press 'q' (key == ord('q')), exit the loop
    # if not (key = cv2.waitKey(1000)), continue
    if key == ord('q'):
        if status == 1:
            times.append(datetime.now())
        break

print(status_list)
print(times)

for i in range(0, len(times), 2):
    df = df.append({"Start":times[i], "End":times[i+1]}, ignore_index = True)

df.to_csv("Times.csv")
video.release()

cv2.destroyAllWindows
