import cv2
import numpy as np
import math

points = []
uv_front = []

Z = 1
count = 1

class Projection(object):

    def __init__(self, image_path, points):
        """
            :param points: Selected pixels on top view(BEV) image
        """

        if type(image_path) != str:
            self.image = image_path
        else:
            self.image = cv2.imread(image_path)
        self.height, self.width, self.channels = self.image.shape

    def top_to_front(self, theta=0, phi=0, gamma=0, dx=0, dy=0, dz=0, fov=90):
        """
            Project the top view pixels to the front view pixels.
            :return: New pixels on perspective(front) view image
        """
        
        #######Projection Algorithm#######
        global count
        f = float(self.width/2*(1/math.tan(fov/180*math.pi/2)))
        T = np.array([[1,0,0,0],[0,0,-1,0],[0,1,0,-1.5],[0,0,0,1]])
        
        for i in range(0,count):
            x_bev = Z*(points[i][0]-256)/f
            y_bev = Z*(points[i][1]-256)/f
            #xy_bev.append([x_bev, y_bev])
            
            xyz_front = np.dot(T, [[x_bev],[y_bev],[Z],[1]])
            xyz_front = np.transpose(xyz_front).tolist()
            
            u_front = int(f*xyz_front[0][0]/xyz_front[0][2]*(-1)+256)
            v_front = int(f*xyz_front[0][1]/xyz_front[0][2]+256)
            
            uv_front.append([u_front, v_front])
        
        ###print the result###
        print("f = ", round(f, 2))
        print("Pick point: ", points)
        print("Projection point: ", uv_front)
        
        new_pixels = uv_front

        return new_pixels

    def show_image(self, new_pixels, img_name='projection.png', color=(0, 0, 255), alpha=0.4):
        """
            Show the projection result and fill the selected area on perspective(front) view image.
        """

        new_image = cv2.fillPoly(
            self.image.copy(), [np.array(new_pixels)], color)
        new_image = cv2.addWeighted(
            new_image, alpha, self.image, (1 - alpha), 0)
        
        cv2.imshow(
            f'Top to front view projection {img_name}', new_image)
        cv2.imwrite(img_name, new_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return new_image


def click_event(event, x, y, flags, params):
    global count
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:

        count = count + 1
        
        #print(x, ' ', y)
        points.append([x, y])
        font = cv2.FONT_HERSHEY_SIMPLEX
        #cv2.putText(img, str(x) + ',' + str(y), (x+5, y+5), font, 0.5, (0, 0, 255), 1)
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow('image', img)

    # checking for right mouse clicks
    if event == cv2.EVENT_RBUTTONDOWN:

        print(x, ' ', y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = img[y, x, 0]
        g = img[y, x, 1]
        r = img[y, x, 2]
        # cv2.putText(img, str(b) + ',' + str(g) + ',' + str(r), (x, y), font, 1, (255, 255, 0), 2)
        cv2.imshow('image', img)

if __name__ == "__main__":

    pitch_ang = -90
    count = 0

    front_rgb = "Front_view.png"
    top_rgb = "BEV_view.png"

    # click the pixels on window
    img = cv2.imread(top_rgb, 1)
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)

    projection = Projection(front_rgb, points)
    new_pixels = projection.top_to_front(theta=pitch_ang)
    projection.show_image(new_pixels)
