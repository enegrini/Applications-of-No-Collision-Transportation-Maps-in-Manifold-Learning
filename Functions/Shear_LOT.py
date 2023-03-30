import numpy as np

class Shear(object):
    def __init__(self):
        pass
        
    def create_shear(self, angle=45, lambda_1=1.2, lambda_2=0.8,
                         shift=[ [0], [-1] ], center=[[13], [13]]):
        # define params for shearing matrix
        self.angle = angle
        self.theta = np.radians(angle)
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.shift = np.asarray(shift)
        self.center = np.asarray(center) # get the center of image
        # create shear
        self.create_matrices()
        
    def create_matrices(self):
        self.P = np.asarray([[np.cos(self.theta), -np.sin(self.theta)],
                        [np.sin(self.theta), np.cos(self.theta)]]) # orthonormal basis
        self.Lambda = np.diag([self.lambda_1, self.lambda_2])
        self.Lambda_inv = np.diag([1/self.lambda_1, 1/self.lambda_2])
        self.A = np.matmul(np.transpose(self.P),
                            np.matmul(self.Lambda,self.P)) # create shear
        # define inverse shear
        self.A_inverse = np.matmul(np.transpose(self.P),
                              np.matmul(self.Lambda_inv, self.P))
      
        #self.b = self.center + self.shift

    def apply_inverse(self, input_coord):
        #inverse_point1 = self.A_inverse @ (input_coord - self.center- self.shift ) + self.center 
        inverse_point = np.matmul(self.A_inverse, (input_coord - self.center - self.shift)) + self.center 
        #print(inverse_point-inverse_point1)
        x = inverse_point[0][0]
        y = inverse_point[1][0]
        return inverse_point, x, y
        
    def find_inverse_point(self, i, j, imag):
        input_coord = np.asarray([ [i], [j] ]) # get current input coordinate
        # get the point that gets mapped to input_coord from applying A
        # we use A_inverse here to get that
        original_point, x, y = self.apply_inverse(input_coord)
        if x > 27 or y > 27 or x<0 or y<0:
            return 0 # original point is outside of grid
        # get points on grid that are close to original inverse_point
        x_floor = np.floor(x)
        x_ceil = np.ceil(x)
        y_floor = np.floor(y)
        y_ceil = np.ceil(y)
        point_1 = np.asarray([ [x_floor], [y_floor] ])
        point_2 = np.asarray([ [x_floor], [y_ceil] ])
        point_3 = np.asarray([ [x_ceil], [y_floor] ])
        point_4 = np.asarray([ [x_ceil], [y_ceil] ])
        all_points = [point_1, point_2, point_3, point_4]
        points = []
        for point in all_points:
            bool_val = any([np.array_equal(point,point_prime) for point_prime in points])
            if not bool_val:
                points.append(point)
        # get rid of repeat points (happens if x or y is already an integer)
        def fractional_value(point_val):
            x_temp = point_val[0][0]
            y_temp = point_val[1][0]
            if x_temp > 27 or y_temp > 27 or x_temp < 0 or y_temp < 0:
                pixel = 0
            else:
                pixel = imag[int(x_temp), int(y_temp)]
            # compute similariy as e^{-||x - y||_2}
            similarity_metric = np.exp(-np.linalg.norm(original_point-point_val))
            return pixel, similarity_metric
        pixel_value = 0
        total_dist = 0
        for point in points:
            imag_val, similarity_metric = fractional_value(point)
            if imag_val > 0:
                total_dist += similarity_metric
            pixel_value += imag_val*similarity_metric
        if total_dist > 0:
            final_pixel_val = pixel_value / total_dist
        else:
            final_pixel_val = pixel_value
        return final_pixel_val

    def shear_image(self,image):
        sheared_image = np.zeros((28,28))
        for i in range(28):
            for j in range(28):
                pixel_ij = self.find_inverse_point(i, j, image)
                sheared_image[i,j] = pixel_ij
        
        return sheared_image

