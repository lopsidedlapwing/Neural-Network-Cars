import random
import os
import math
import numpy as np
import time
import sys
from datetime import datetime
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from operator import attrgetter

from scipy.special import expit # Sigmoid function that doesn't throw rounding errors
from copy import deepcopy # To quickly copy weight arrays without changing parents
from gamecode import * # Store files not directly related to neural network in separate file
from more_itertools import pairwise # For filling bias and weight arrays from lists
from scipy.stats import truncnorm # Random variable normally distributed within bounds
from scipy.stats import poisson # Distribute number of mutations around mutationRate

# ---- Global neural net variables ---- #

inputLayerSize = 6  # Number of input vertices - five distance lines and current speed
hiddenLayerSize = 6 # One hidden layer with six vertices/neurons
outputLayerSize = 4 # Accelerate, brake, right, left
layerSizes = [inputLayerSize, hiddenLayerSize, outputLayerSize]
number_of_weights = sum([a * b for a, b in pairwise(layerSizes)])
number_of_biases = sum(layerSizes[1:])

# ---- Neural net functions ---- #

def list_from_array(input_array): # Turns ND array of values into 1D list of values
    return np.concatenate(deepcopy(input_array)).ravel()

def array_from_list(input_list, target_array): # List of values back to array of values, needs car_property to know array shape
    shapes = [arr.shape for arr in target_array] # Shape of each numpy array in property
    counts = [0] + [x*y for x, y in shapes] # Number of elements in each numpy array, plus zero for indexing
    index_pairs = pairwise(np.cumsum(counts)) # Slices of list corresponding to each array
    return [input_list[a:b].reshape(shape) for (a, b), shape in zip(index_pairs, shapes)] # Sets values into array

def modify_n_randomly(values_array, n=1): # Mutates the given values_array (weights or biases) n times
    values_list = list_from_array(values_array) # Turns values array into list
    indices = random.choices(range(len(values_list)), k=n) # Chooses n random indices of list
    coefficients = [m + 1 for m in truncnorm.rvs(-0.9, 0.9, size=n)] # n random numbers, normal distribution between 0 and 2
    for index, coefficient in zip(indices, coefficients):
        values_list[index] = values_list[index] * coefficient # Changes values in list
    return array_from_list(values_list, values_array) # Reads modified list values back into array

# ---- Game functions ---- #

def move_point(point, angle, unit): #Translate a point in a given direction
    x, y = point
    rad = math.radians(-angle % 360)
    x += unit * math.sin(rad)
    y += unit * math.cos(rad)
    return x, y

def rotation(origin, point, angle): #Used to rotate points #rotate(origin, point, math.radians(10))
    ox, oy = origin
    px, py = point
    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

def displayTexts(): # Info text displayed on game window, hide with 'd'
    global selectedCars

    infoX = 1365
    infoY = 600
    font = pygame.font.Font('freesansbold.ttf', 18)
    text1 = font.render('0..9 - Change Mutation', True, white)
    text2 = font.render('LMB - Select/Unselect', True, white)
    text3 = font.render('RMB - Delete', True, white)
    text4 = font.render('L - Show/Hide Lines', True, white)
    text5 = font.render('R - Reset', True, white)
    text6 = font.render('B - Breed', True, white)
    text7 = font.render('C - Clean', True, white)
    text8 = font.render('N - Next Track', True, white)
    text9 = font.render('A - Toggle Player', True, white)
    text10 = font.render('D - Toggle Info', True, white)
    text11 = font.render('M - Breed and Next Track', True, white)
    text1Rect = text1.get_rect().move(infoX, infoY)
    texts = [text1, text2, text3, text4, text5, text6, text7, text8, text9, text10, text11]
    rects = [text.get_rect().move(infoX, infoY + index*text1Rect.height) for index, text in enumerate(texts)]

    infotextX = 20
    infotextY = 600
    infotext1 = font.render('Gen ' + str(generation), True, white)
    infotext2 = font.render('Cars: ' + str(num_of_nnCars), True, white)
    infotext3 = font.render('Alive: ' + str(alive), True, white)
    infotext4 = font.render('Selected: ' + str(len(selectedCars)), True, white)
    infotext5 = font.render('Lines ON', True, white) if lines else font.render('Lines OFF', True, white)
    infotext6 = infotext6 = font.render('Player ON', True, white) if player else font.render('Player OFF', True, white)
    infotext7 = font.render('Mutation: '+ str(mutationRate), True, white)
    infotext8 = font.render('Frames: ' + str(frames), True, white)
    infotext9 = font.render('FPS: ' + str(FPS), True, white)
    infotext1Rect = infotext1.get_rect().move(infotextX, infotextY)
    infotexts = [infotext1, infotext2, infotext3, infotext4, infotext5, infotext6, infotext7, infotext8, infotext9]
    inforects = [text.get_rect().move(infotextX, infotextY + index*infotext1Rect.height) for index, text in enumerate(infotexts)]

    for text, rect in zip(texts + infotexts, rects + inforects):
        gameDisplay.blit(text, rect)

def redrawGameWindow(): #Called on every frame
    global alive
    global frames
    global img
    global is_paused
    global player_car
    if is_paused:
        return
    frames += 1
    gameD = gameDisplay.blit(bg, (0, 0))

    #NN cars
    for nncar in nnCars:
        if not nncar.collided:
            nncar.update() #Update: Every car center coord, corners, directions, collision points and collision distances

        if nncar.collision(): #Check which car collided
            nncar.collided = True #If collided then change collided attribute to true
            if nncar.yaReste == False:
                alive -= 1
                nncar.yaReste = True

        else: #If not collided then feedforward the input and take an action
            nncar.feedforward()
            nncar.takeAction()
        nncar.draw(gameDisplay)

    #Same but for player
    if player:
        player_car.update()
        if player_car.collision():
            player_car.resetPosition()
            player_car.update()
        player_car.draw(gameDisplay)

    if display_info:
        displayTexts()

    pygame.display.update()

# ---- Car class ---- #

class Car:
    def __init__(self, sizes):
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]] #Biases
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])] #Weights
        #self.biases = [np.random.uniform(size=(y, 1)) for y in sizes[1:]] #Biases
        #self.weights = [np.random.uniform(size=(y, x)) for x, y in zip(sizes[:-1], sizes[1:])] #Weights
        self.collision_points = [(0, 0)] * 5 # Points at which car could collide
        self.distances = [0] * 5 # Distances to above points, drawn in-game as red lines
        self.angle_offsets = [0, 45, -45, 90, -90] # Relative angles of collision lines
        self.yaReste = False # Has the car collided?
        self.collided = False
        self.inp = np.array([[dist] for dist in self.distances]) # Input to neural network
        self.outp = np.array([[0], [0], [0], [0]]) # Output from neural network
        self.showlines = False #Boolean used for toggling distance lines
        self.x = 120
        self.y = 480
        self.corners = [(self.x, self.y)] * 4
        self.height = 35
        self.width = 17
        self.velocity = 0
        self.acceleration = 0
        self.angle = 180
        self.set_corners()
        self.color = white
        self.car_image = white_small_car

    def set_corners(self):
        self.corners[0] = self.x - self.width/2, self.y - self.height/2
        self.corners[1] = self.x + self.width/2, self.y - self.height/2
        self.corners[2] = self.x + self.width/2, self.y + self.height/2
        self.corners[3] = self.x - self.width/2, self.y + self.height/2
        self.corners = [rotation((self.x, self.y), corner, math.radians(self.angle)) for corner in self.corners]

    def rotate(self, rot):
        self.angle += rot
        self.angle = self.angle % 360

    def is_track(self, point):
        point = int(point[0]), int(point[1])
        return track_image.get_at(point).a != 0

    def update(self): #En cada frame actualizo los vertices (traslacion y rotacion) y los puntos de colision
        self.x, self.y = move_point((self.x, self.y), self.angle, self.velocity)
        self.set_corners()

        if self.acceleration != 0:
            self.velocity += self.acceleration
            if self.velocity > maxspeed:
                self.velocity = maxspeed
            elif self.velocity < 0:
                self.velocity = 0
        else:
            self.velocity *= 0.92 # Friction

        self.collision_points = [(self.x, self.y)] * 5
        new_collision_points = []
        for point, angle_offset in zip(self.collision_points, self.angle_offsets):
            angle = self.angle + angle_offset
            while self.is_track(point):
                point = move_point(point, angle, 10)
            while not self.is_track(point):
                point = move_point(point, angle, -1)
            new_collision_points.append(point)
        self.collision_points = new_collision_points

        self.distances = [int(math.dist((self.x, self.y), (cp[0], cp[1]))) for cp in self.collision_points]

    def draw(self, display):
        rotated_image = pygame.transform.rotate(self.car_image, -self.angle-180)
        rect_rotated_image = rotated_image.get_rect()
        rect_rotated_image.center = self.x, self.y
        gameDisplay.blit(rotated_image, rect_rotated_image)

        if self.showlines:
            [pygame.draw.line(gameDisplay, Color_line, (self.x, self.y), c, 2) for c in self.collision_points]

    def feedforward(self):
        self.inp = np.array([[dist] for dist in self.distances] + [[self.velocity]])
        for b, w in zip(self.biases, self.weights):
            self.inp = expit(np.dot(w, self.inp) + b)
        self.outp = self.inp

    def collision(self):
        has_not_collided = all(map(self.is_track, self.corners))
        return not has_not_collided # Easier to work out if car hasn't collided and return the negation

    def resetPosition(self):
        self.x = 120
        self.y = 480
        self.angle = 180

    def takeAction(self):
        if self.outp.item(0) > 0.5: #Accelerate
            self.acceleration = 0.2
        else:
            self.acceleration = 0
        if self.outp.item(1) > 0.5: #Brake
            self.acceleration = -0.2
        if self.outp.item(2) > 0.5: #Turn right
            self.rotate(-5)
        if self.outp.item(3) > 0.5: #Turn left
            self.rotate(5)

# ---- User input functions ---- #

def show_lines():
    global lines
    global player_car
    player_car.showlines = not player_car.showlines
    lines = not lines

def scrap_stuck_cars():
    global nnCars
    global alive
    nnCars = [car for car in nnCars if not car.collided and car.velocity > 0.001]
    alive = len(nnCars)

def next_map():
    global number_track
    global bg
    global track_image

    number_track += 1
    for nncar in nnCars:
        nncar.velocity = 0
        nncar.acceleration = 0
        nncar.x = 140
        nncar.y = 610
        nncar.angle = 180
        nncar.collided = False
    generateRandomMap(gameDisplay)
    bg = pygame.image.load('randomGeneratedTrackFront.png')
    track_image = pygame.image.load('randomGeneratedTrackBack.png')

def next_gen():
    global selectedCars
    global generation
    global alive
    global nnCars
    global num_of_nnCars

    while len(selectedCars) < 2:
        selectedCars.append(random.choice(nnCars))

    alive = num_of_nnCars
    generation += 1
    nnCars.clear()

    for i in range(num_of_nnCars):
        nnCars.append(Car(layerSizes))

    # Randomly mix parent genomes for each child
    parent1_weights = list_from_array(selectedCars[0].weights)
    parent2_weights = list_from_array(selectedCars[1].weights)
    parent1_biases = list_from_array(selectedCars[0].biases)
    parent2_biases = list_from_array(selectedCars[1].biases)

    for i in range(2, num_of_nnCars-2):
        random_weight_choices = np.random.randint(2, size=number_of_weights) # Generate random number lists
        random_bias_choices = np.random.randint(2, size=number_of_biases)
        cross_weights = np.choose(random_weight_choices, [parent1_weights, parent2_weights]) # Choose values from parents
        cross_biases = np.choose(random_bias_choices, [parent1_biases, parent2_biases])
        nnCars[i].weights = array_from_list(cross_weights, nnCars[i].weights) # Give values to child cars
        nnCars[i].biases = array_from_list(cross_biases, nnCars[i].biases)

    # Add parent cars to next generation
    nnCars[num_of_nnCars-2] = selectedCars[0]
    nnCars[num_of_nnCars-1] = selectedCars[1]

    for car in nnCars[-2:]:
        car.car_image = green_small_car
        car.resetPosition()
        car.collided = False

    # Split mutationRate proportionately between weights and biases
    weight_mutation_rate = math.ceil(mutationRate * number_of_weights/(number_of_weights + number_of_biases))
    bias_mutation_rate = math.ceil(mutationRate * number_of_biases/(number_of_weights + number_of_biases))

    # Modify each genome by a random number of mutations centered on it's split of mutationRate
    weight_mutation_counts = poisson.rvs(weight_mutation_rate, size=num_of_nnCars-2)
    bias_mutation_counts = poisson.rvs(bias_mutation_rate, size=num_of_nnCars-2)
    for index, (weight_mutations, bias_mutations) in enumerate(zip(weight_mutation_counts, bias_mutation_counts)):
        nnCars[index].weights = modify_n_randomly(nnCars[index].weights, n=weight_mutations)
        nnCars[index].biases = modify_n_randomly(nnCars[index].biases, n=bias_mutations)

    for car in nnCars:
        car.x, car.y = (120, 480) if number_track == 1 else (140, 610)

    selectedCars.clear()

def next_gen_and_map():
    global selectedCars
    while len(selectedCars) < 2:
        selectedCars.append(random.choice(nnCars))
    next_gen()
    next_map()

def reload_map():
    global generation
    global alive
    global num_of_nnCars
    global nnCars
    global selectedCars
    generation = 1
    alive = num_of_nnCars
    nnCars.clear()
    selectedCars.clear()
    for i in range(num_of_nnCars):
        nnCars.append(Car(layerSizes))
    for car in nnCars:
        car.x, car.y = (120, 480) if number_track == 1 else (140, 610)

def swap_sprite(car):
    originals = [white_small_car, white_big_car, green_small_car, green_big_car]
    replacements = [white_big_car, white_small_car, green_big_car, green_small_car]
    car.car_image = replacements[originals.index(car.car_image)]
    return car

def clear_selected_cars():
    global selectedCars
    global nnCars
    for car in selectedCars:
        car = swap_sprite(car)
    selectedCars.clear()

def mouseclick():
    global alive
    global nnCars
    global selectedCars
    global is_paused
    mouses = pygame.mouse.get_pressed()
    if mouses[0]: # Left click, choose parent cars
        pos = pygame.mouse.get_pos()
        point = Point(pos[0], pos[1])
        for nncar in nnCars:
            polygon = Polygon(nncar.corners)
            if (polygon.contains(point)): # If car has been clicked
                if nncar in selectedCars: # If it's a selected car, deselect it
                    selectedCars.remove(nncar)
                    nncar = swap_sprite(nncar)
                    if nncar.collided:
                        nncar.velocity = 0
                        nncar.acceleration = 0
                    if not is_paused:
                        nncar.update()
                else: # If it's an unselected car, select it
                    if len(selectedCars) < 2:
                        selectedCars.append(nncar)
                        nncar = swap_sprite(nncar)
                        if nncar.collided:
                            nncar.velocity = 0
                            nncar.acceleration = 0
                        if not is_paused:
                            nncar.update()
                break # Only affect one car per click

    if mouses[2]: # Right click, remove cars
        pos = pygame.mouse.get_pos()
        point = Point(pos[0], pos[1])
        for nncar in nnCars:
            polygon = Polygon(nncar.corners)
            if (polygon.contains(point)):
                if nncar not in selectedCars:
                    nnCars.remove(nncar)
                    if not nncar.collided:
                        alive -= 1
                break # Only affect one car per click

def swap_player():
    global player
    player = not player

def swap_display():
    global display_info
    display_info = not display_info

def change_mutation_rate(input):
    global mutationRate
    mutationRate = 10 * (input - ord('0'))

def choose_cars():
    global nnCars
    global selectedCars
    if len(selectedCars) == 2:
        for car in selectedCars:
            car = swap_sprite(car)
        selectedCars.clear()

    while len(selectedCars) < 2:
        chosen_car = random.choice(nnCars)
        if chosen_car in selectedCars:
            continue
        selectedCars.append(chosen_car)
        chosen_car = swap_sprite(chosen_car)

def pause_game():
    global is_paused
    is_paused = not is_paused

# ---- Player input reaction ---- #

def key_event():
    mappings = [
        ('l', show_lines),
        ('c', scrap_stuck_cars),
        ('a', swap_player),
        ('d', swap_display),
        ('n', next_map),
        ('z', clear_selected_cars),
        ('b', next_gen),
        ('m', next_gen_and_map),
        ('r', reload_map),
        ('g', choose_cars),
        ('p', pause_game)
    ]
    [func() for (key, func) in mappings if ord(key) == event.key]
    if ord('0') <= event.key and event.key <= ord('9'):
        change_mutation_rate(event.key)

# ---- Global car variables ---- # These can't go at the top since they need the Car class to be defined first

player_car = Car(layerSizes) # Car instance driven by player, technically a global variable but must come after Car definition
for i in range(num_of_nnCars): # Populate game with neural net-driven cars
    nnCars.append(Car(layerSizes))

# ---- Game loop ---- #

while True:
    for event in pygame.event.get(): #Check for events
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

        if event.type == pygame.KEYDOWN: #If user uses the keyboard
            key_event()

        if event.type == pygame.MOUSEBUTTONDOWN:
            mouseclick()

    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        player_car.rotate(-5)
    if keys[pygame.K_RIGHT]:
        player_car.rotate(5)
    if keys[pygame.K_UP]:
        player_car.acceleration = 0.2
    else:
        player_car.acceleration = 0
    if keys[pygame.K_DOWN]:
        player_car.acceleration = -0.2

    redrawGameWindow()
    clock.tick(FPS)
