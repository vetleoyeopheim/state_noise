import random
import numpy as np
import math
from scipy.ndimage import gaussian_filter as gaussf
from numba import njit


class TwoStateNoise():


    def __init__(self, x, y, sigma, prob):
        """
        Generate a noise map based on two states, ocean(0) and land(1)
        The algorithm starts with a 2d array of 0
        Then the state of each element is determined by the state of previous elements in the x and y direction, and given probabilities
        Finally the map is put through a gaussian blur filter with a given sigma
        prob is a list of 4 probabilities for the state being 0 for 4 different cases of neighbouring elements : [prob_x0_y0, prob_x1_y0, prob_x0_y1, prob_x1_y1]

        """

        self.x = x
        self.y = y
        self.sigma = sigma
        self.prob = prob



    def gen_twostate_noisemap(self, amap = []):
        """
        If new_map is an empty list then new map is generated from scratch
        If new_map is a map then the map provided is put through another iteration of the algorithm
        
        """
        if len(amap) == 0:
            n_map = np.zeros((self.y, self.x))
        else:
            n_map = amap
        
        #Call a function using numba to cut down on computation time
        n_map = twostate_loop(n_map, self.y, self.x, self.prob)

        n_map = gaussf(n_map, sigma = self.sigma)

        n_map = np.arctan(n_map)

        #Normalize noise map
        n_map = self.normalize_map(n_map)

        return n_map

    def normalize_map(self, noise_map):
        map_norm = (noise_map - noise_map.min()) / (noise_map.max() - noise_map.min())
        return map_norm



@njit
def twostate_loop(init_map, nx, ny, prob):

    n_map = init_map

    for i in range(1,nx,1):
        for j in range(1,ny,1):
            if n_map[i - 1][j] == 0 and n_map[i][j - 1] == 0:
                prob_zero = prob[0]       #prob_zero is the probability that the element will be in state zero
            elif n_map[i - 1][j] == 1 and n_map[i][j - 1] == 0:
                prob_zero = prob[1]
            elif n_map[i - 1][j] == 0 and n_map[i][j - 1] == 1:
                prob_zero = prob[2]
            elif n_map[i - 1][j] == 1 and n_map[i][j - 1] == 1:
                prob_zero = prob[3]

            rand_draw = np.random.random()
            if prob_zero > rand_draw:
                n_map[i][j] = 0
            else:
                n_map[i][j] = 1

    return n_map

