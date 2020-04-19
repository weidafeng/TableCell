

# =============================================================================
# 点类
# =============================================================================
class Point:
    def __init__(self, point):
        # x1, y1, x2, y2 = l # 前两个数为起点，后两个数为终点
        self.x = point[0]
        self.y = point[1]

    def copy(self):
        return self

    def toList(self):
        # 将点类转化为list类型
        return [int(self.x), int(self.y)]

    def lenth(self):
        return 1. * (self.x * self.x + self.y * self.y) ** 0.5

    def measureAngle(self, lastPoint, nextPoint):
        # 计算尖锐度，参考 https://www.cnblogs.com/jsxyhelu/p/5106760.html
        vect1 = [self.x - lastPoint.x, self.y - lastPoint.y]

        vect2 = [self.x - nextPoint.x, self.y - nextPoint.y]

        vect3 = [lastPoint.x - nextPoint.x, lastPoint.y - nextPoint.y]

        sin = 1.0 * Point(vect3).lenth() / (Point(vect1).lenth() + Point(vect2).lenth())
        return 1 - sin

    def printf(self):
        print((self.x, self.y))


# =============================================================================
# 轮廓类
# =============================================================================
class Contour(Point):
    def __init__(self, contour):
        self.contour = []
        for p in contour:
            self.contour.append(Point(p[0]))
        self.length = len(contour)

    def pickLeftPoint(self, currentLocation, setp):
        # 防止取左边相邻点时越界
        if currentLocation - setp < 0:
            # print(currentLocation-setp+self.length)
            return currentLocation - setp + self.length
        else:
            # print(currentLocation-setp)
            return currentLocation - setp

    def pickRightPoint(self, currentLocation, setp):
        # 防止取右边相邻点时越界
        if currentLocation + setp > self.length - 1:
            # print(currentLocation+setp-self.length+1)
            return currentLocation + setp - self.length + 1
        else:
            # print(currentLocation+setp)
            return currentLocation + setp

    def getAngle(self, p, setp):
        # print(p)
        return self.contour[p].measureAngle(self.contour[self.pickRightPoint(p, setp)],
                                            self.contour[self.pickLeftPoint(p, setp)])


def sortPoint(rowdata):
    x = 0
    y = 0
    for p in rowdata:
        x = p.x + x
        y = p.y + y
    x = x / 4
    y = y / 4
    sorteddata = [[0, 0]] * 4
    for p in rowdata:
        if p.x < x and p.y < y:
            sorteddata[0] = p.toList()
        if p.x > x and p.y < y:
            sorteddata[1] = p.toList()
        if p.x > x and p.y > y:
            sorteddata[2] = p.toList()
        if p.x < x and p.y > y:
            sorteddata[3] = p.toList()
    return sorteddata


def getPoint(contours):
    index = 0
    contour = contours[1]
    j = 0
    size = 0
    for i in contour:
        if i.size > size:
            size = i.size
            index = j
        j = j + 1
    maxContour = Contour(contour[index])
    data = []
    datas = []
    for p in range(0, maxContour.length - 1):
        y = maxContour.getAngle(p, 5)
        datas.append(y)
        if 0.1 < y:
            data.append(maxContour.contour[p])
    plt.plot(datas)
    plt.show()


if __name__ == '__main__':
    old_img = cv2.imread('1.jpg')
    t_points = img_process(old_img)

