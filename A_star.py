import matplotlib.pyplot as plt
import matplotlib.axes as Axes
from datetime import datetime
import math
import cv2
import numpy as np
import imutils

show_animation = True

class Node:

    def __init__(self, x, y, cost, pind):
        self.x = x
        self.y = y
        self.cost = cost
        self.pind = pind

    def __str__(self):
        return str(self.x) + "," + str(self.y) + "," + str(self.cost) + "," + str(self.pind)


def calc_final_path(ngoal, closedset, reso):
    # generate final course
    rx, ry = [ngoal.x * reso], [ngoal.y * reso]
    pind = ngoal.pind
    while pind != -1:
        n = closedset[pind]
        rx.append(n.x * reso)
        ry.append(n.y * reso)
        pind = n.pind

    return rx, ry


def a_star_planning(sx, sy, gx, gy, ox, oy, reso, rr):
    """
    gx: goal x position [m]
    gx: goal x position [m]
    ox: x position list of Obstacles [m]
    oy: y position list of Obstacles [m]
    reso: grid resolution [m]
    rr: robot radius[m]
    """

    nstart = Node(round(sx / reso), round(sy / reso), 0.0, -1)
    ngoal = Node(round(gx / reso), round(gy / reso), 0.0, -1)
    ox = [iox / reso for iox in ox]
    oy = [ioy / reso for ioy in oy]

    obmap, minx, miny, maxx, maxy, xw, yw = calc_obstacle_map(ox, oy, reso, rr)

    motion = get_motion_model()

    openset, closedset = dict(), dict()
    openset[calc_index(nstart, xw, minx, miny)] = nstart
    while 1:
        c_id = min(
            openset, key=lambda o: openset[o].cost + calc_heuristic(ngoal, openset[o]))
        current = openset[c_id]

        if current.x == ngoal.x and current.y == ngoal.y:
            print("Find goal")
            ngoal.pind = current.pind
            ngoal.cost = current.cost
            break

        # Remove the item from the open set
        del openset[c_id]
        # Add it to the closed set
        closedset[c_id] = current

        # expand search grid based on motion model
        for i in range(len(motion)):
            node = Node(current.x + motion[i][0],
                        current.y + motion[i][1],
                        current.cost + motion[i][2], c_id)
            n_id = calc_index(node, xw, minx, miny)

            if n_id in closedset:
                continue

            if not verify_node(node, obmap, minx, miny, maxx, maxy):
                continue

            if n_id not in openset:
                openset[n_id] = node  # Discover a new node
            else:
                if openset[n_id].cost >= node.cost:
                    # This path is the best until now. record it!
                    openset[n_id] = node

    rx, ry = calc_final_path(ngoal, closedset, reso)

    return rx, ry


def calc_heuristic(n1, n2):
    w = 1.0  # weight of heuristic
    d = w * math.sqrt((n1.x - n2.x)**2 + (n1.y - n2.y)**2)
    return d


def verify_node(node, obmap, minx, miny, maxx, maxy):

    if node.x < minx:
        return False
    elif node.y < miny:
        return False
    elif node.x >= maxx:
        return False
    elif node.y >= maxy:
        return False

    if obmap[node.x][node.y]:
        return False

    return True


def calc_obstacle_map(ox, oy, reso, vr):

    minx = round(min(ox))
    miny = round(min(oy))
    maxx = round(max(ox))
    maxy = round(max(oy))

    xwidth = round(maxx - minx)
    ywidth = round(maxy - miny)

    # obstacle map generation
    obmap = [[False for i in range(xwidth)] for i in range(ywidth)]
    for ix in range(xwidth):
        x = ix + minx
        for iy in range(ywidth):
            y = iy + miny
            #  print(x, y)
            for iox, ioy in zip(ox, oy):
                d = math.sqrt((iox - x)**2 + (ioy - y)**2)
                if d <= vr / reso:
                    obmap[ix][iy] = True
                    break

    return obmap, minx, miny, maxx, maxy, xwidth, ywidth


def calc_index(node, xwidth, xmin, ymin):
    return (node.y - ymin) * xwidth + (node.x - xmin)


def get_motion_model():
    # dx, dy, cost
    motion = [[1, 0, 1],
              [0, 1, 1],
              [-1, 0, 1],
              [0, -1, 1],
              [-1, -1, math.sqrt(2)],
              [-1, 1, math.sqrt(2)],
              [1, -1, math.sqrt(2)],
              [1, 1, math.sqrt(2)]]

    return motion

def frmt(cnts):
    cnt = []
    for c in cnts:
        x = []
        for y in c:
            l = [y[0][0],y[0][1]]
            x.append(l)
        cnt.append(x)
    return cnt

def calculate_contour(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    cv2.bitwise_not(thresh, thresh)

    cnts = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    i =  0 if imutils.is_cv2() else 1
    cnts = cnts[i]
    return cnts
    c = min(cnts, key=cv2.contourArea)
    return cnts

def pathlength(x,y):
    n = len(x) 
    lv = [math.sqrt((x[i]-x[i-1])**2 + (y[i]-y[i-1])**2) for i in range (1,n)]
    L = sum(lv)
    return L

def main():
    print(__file__ + " start!!")

    # start and goal position
    sx = 4.0  # [m]
    sy = 4.0  # [m]
    gx = 127.0  # [m]
    gy = 107.0  # [m]
    grid_size = 1.0  # [m]
    robot_size = 1.0  # [m]

    ox, oy = [], []

    for i in range(200):
        ox.append(i)
        oy.append(0.0)
    for i in range(200):
        ox.append(200.0)
        oy.append(i)
    for i in range(200):
        ox.append(i)
        oy.append(200.0)
    for i in range(200):
        ox.append(0.0)
        oy.append(i)
    img = cv2.imread('pic1.jpg')
    img = cv2.resize(img,(1000,1000), interpolation = cv2.INTER_AREA)
    img = cv2.resize(img,(200,200), interpolation = cv2.INTER_AREA)
    # cv2.imshow("1",img)
    # cv2.waitKey(0)
    cnts = calculate_contour(img)
    cnts = frmt(cnts)
    for c in cnts:
        for m in c:
            ox.append(m[0])
            oy.append(199-m[1])

    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    if show_animation:
        ax.plot(ox, oy, ".k")
        ax.plot(sx, sy, "xr")
        for c in cnts:
            if len(c) > 1:
                ax.contourf(c)
        ax.plot(gx, gy, "xb")
        ax.grid(True)
        # plt.axis("scaled")
        ax.set_xlim(-50, 250)
        ax.set_ylim(-50, 250)
        ax.axhline(1, color='black', lw=2)
    startTime = datetime.now()
    rx, ry = a_star_planning(sx, sy, gx, gy, ox, oy, grid_size, robot_size)
    print("Length of the optimal path:- "+ str(pathlength(rx,ry)))
    print("Time taken by the program:- " + str(datetime.now() - startTime))
    if show_animation:
        ax.plot(rx, ry, "-r")
        plt.show()
        # plt.savefig("res.jpg")    
    plt.show()

if __name__ == '__main__':
    main()
