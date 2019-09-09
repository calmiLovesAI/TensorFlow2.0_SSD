
class IoU():
    def __init__(self, coord_pred, coord_true):
        super(IoU, self).__init__()
        self.coord_pred = coord_pred
        self.coord_true = coord_true

    def __is_the_point_on_the_line_segment(self, point, segment):
        if max(segment) > point > min(segment):
            return True
        else:
            return False

    def __intersection_of_two_segments(self, a_1, a_2, b_1, b_2):
        a_list = [a_1, a_2]
        b_list = [b_1, b_2]
        a_min = min(a_list)
        a_max = max(a_list)
        b_min = min(b_list)
        b_max = max(b_list)
        if self.__is_the_point_on_the_line_segment(a_min, b_list):
            if self.__is_the_point_on_the_line_segment(a_max, b_list):
                intersection_list = [a_min, a_max]
            else:
                intersection_list = [a_min, b_max]
        else:
            if self.__is_the_point_on_the_line_segment(a_max, b_list):
                intersection_list = [b_min, a_max]
            else:
                if self.__is_the_point_on_the_line_segment(b_min, a_list):
                    intersection_list = [b_min, b_max]
                else:
                    intersection_list = []

        return intersection_list

    def __rectangle_area(self, x1, x2, y1, y2):
        area = abs(x1 - x2) * abs(y1 - y2)
        return area

    def calculate_iou(self):
        # coord_pred : predicted anchors coordinates, [xmin, ymin, xmax, ymax]
        # coord_true : true anchors coordinates, [xmin, ymin, xmax, ymax]
        x_intersection = self.__intersection_of_two_segments(self.coord_pred[0], self.coord_pred[2], self.coord_true[0], self.coord_true[2])
        y_intersection = self.__intersection_of_two_segments(self.coord_pred[1], self.coord_pred[3], self.coord_true[1], self.coord_true[3])

        if x_intersection and y_intersection:
            i_area = self.__rectangle_area(x1=x_intersection[0],
                                           x2=x_intersection[1],
                                           y1=y_intersection[0],
                                           y2=y_intersection[1])
        else:
            i_area = 0

        u_area = self.__rectangle_area(x1=self.coord_pred[0], x2=self.coord_pred[2],
                                       y1=self.coord_pred[1], y2=self.coord_pred[3]) + \
                 self.__rectangle_area(x1=self.coord_true[0], x2=self.coord_true[2],
                                       y1=self.coord_true[1], y2=self.coord_true[3]) - \
            i_area

        iou_area = i_area / u_area

        return iou_area
