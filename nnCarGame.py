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

# ---- Global neural net variables ---- #

inputLayerSize = 6  # Number of input vertices - five distance lines and current speed
hiddenLayerSize = 6 # One hidden layer with six vertices/neurons
outputLayerSize = 4 # Accelerate, brake, right, left
layerSizes = [inputLayerSize, hiddenLayerSize, outputLayerSize]
sizenn = len(layerSizes) # Number of layers in neural network

# ---- Neural net functions ---- #

def sigmoid(z): #Sigmoid function, used as the neurons activation function
    return expit(z) # np.exp gives warning errors when rounding

def modify_one_randomly(input_list):
    index = random.randint(0, len(input_list)-1)
    input_list[index] = input_list[index] * random.uniform(0.8, 1.2)
    return input_list

def swap_alternate(a_in, b_in): # list[::2] iterates every other element of list
    a = deepcopy(a_in)
    b = deepcopy(b_in)
    a[::2] = b_in[::2]
    b[::2] = a_in[::2]
    return a, b

def copy_weights(parent, child):
    child.weights = deepcopy(parent.weights)
    return child

def copy_biases(parent, child):
    child.biases = deepcopy(parent.biases)
    return child

def get_weights_list(child):
    w = deepcopy(child.weights)
    return np.concatenate(w).ravel() # ravel turns arrays into 1D equivalents

def get_biases_list(child):
    b = deepcopy(child.biases)
    return np.concatenate(b).ravel()

def copy_weights_genome(genomeWeights, child):
    child.weights[0] = genomeWeights[:36].reshape((6, 6))
    child.weights[1] = genomeWeights[36:].reshape((4, 6))
    return child

def copy_biases_genome(genomeBiases, child):
    child.biases[0] = genomeBiases[:6].reshape((6, 1))
    child.biases[1] = genomeBiases[6:].reshape((4, 1))
    return child

def mutateOneWeightGene(parent, child):
    child = copy_biases(parent, child)
    genomeWeights = get_weights_list(parent)
    genomeWeights = modify_one_randomly(genomeWeights)
    child = copy_weights_genome(genomeWeights, child)
    return parent, child

def mutateOneBiasesGene(parent, child):
    child = copy_weights(parent, child)
    genomeBiases = get_biases_list(parent)
    genomeBiases = modify_one_randomly(genomeBiases)
    child = copy_biases_genome(genomeBiases, child)
    return parent, child

def uniformCrossOverWeights(parent1, parent2, child1, child2): #Given two parent car objects, it modifies the children car objects weights
    child1 = copy_biases(parent1, child1)
    child2 = copy_biases(parent2, child2)
    genome1 = get_weights_list(parent1)
    genome2 = get_weights_list(parent2)
    genome1, genome2 = swap_alternate(genome1, genome2)
    child1 = copy_weights_genome(genome1, child1)
    child2 = copy_weights_genome(genome2, child2)
    return child1, child2

def uniformCrossOverBiases(parent1, parent2, child1, child2): #Given two parent car objects, it modifies the children car objects biases
    child1 = copy_weights(parent1, child1)
    child2 = copy_weights(parent2, child2)
    genome1 = get_biases_list(parent1)
    genome2 = get_biases_list(parent2)
    genome1, genome2 = swap_alternate(genome1, genome2)
    child1 = copy_biases_genome(genome1, child1)
    child2 = copy_biases_genome(genome2, child2)
    return child1, child2

# ---- Game functions ---- #

class Car:
    def __init__(self, sizes):
        self.score = 0
        self.num_layers = len(sizes) #Number of nn layers
        self.sizes = sizes #List with number of neurons per layer
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]] #Biases
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])] #Weights 
        self.collision_points = [(0, 0)] * 5 # Points at which car could collide
        self.distances = [0] * 5 # Distances to above points, drawn in-game as red lines
        self.yaReste = False
        #The input and output of the NN must be in a numpy array format
        self.inp = np.array([[dist] for dist in self.distances])
        self.outp = np.array([[0], [0], [0], [0]])
        #Boolean used for toggling distance lines
        self.showlines = False
        #Initial location of the car
        self.x = 120
        self.y = 480
        self.center = self.x, self.y
        #Height and width of the car
        self.height = 35 #45
        self.width = 17 #25
        #These are the four corners of the car, using polygon instead of rectangle object, when rotating or moving the car, we rotate or move these
        self.d = self.x - (self.width/2), self.y - (self.height/2)
        self.c = self.x + (self.width/2), self.y - (self.height/2)
        self.b = self.x + (self.width/2), self.y + (self.height/2) #El rectangulo está centrado en (x, y)
        self.a = self.x - (self.width/2), self.y + (self.height/2)              #(a), (b), (c), (d) son los vertices
        #Velocity, acceleration and direction of the car
        self.velocity = 0
        self.acceleration = 0  
        self.angle = 180
        #Boolean which goes true when car collides
        self.collided = False
        #Car color and image
        self.color = white
        self.car_image = white_small_car

    def set_accel(self, accel): 
        self.acceleration = accel

    def rotate(self, rot): 
        self.angle += rot
        self.angle = self.angle % 360

    def is_track(self, point):
        point = int(point[0]), int(point[1])
        return track_image.get_at(point).a != 0

    def update(self): #En cada frame actualizo los vertices (traslacion y rotacion) y los puntos de colision
        def calculateDistance(x1, y1, x2, y2): #Used to calculate distance between points
            return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        def rotation(origin, point, angle): #Used to rotate points #rotate(origin, point, math.radians(10))
            ox, oy = origin
            px, py = point
            qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
            qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
            return qx, qy

        def move_point(point, angle, unit): #Translate a point in a given direction
            x, y = point
            rad = math.radians(-angle % 360)
            x += unit * math.sin(rad)
            y += unit * math.cos(rad)
            return x, y

        self.score += self.velocity
        if self.acceleration != 0:
            self.velocity += self.acceleration
            if self.velocity > maxspeed:
                self.velocity = maxspeed
            elif self.velocity < 0:
                self.velocity = 0
        else:
            self.velocity *= 0.92

        self.x, self.y = move_point((self.x, self.y), self.angle, self.velocity)
        self.center = self.x, self.y

        self.d = self.x - (self.width/2), self.y - (self.height/2) # Get corner positions from centre as if car wasn't rotated
        self.c = self.x + (self.width/2), self.y - (self.height/2)
        self.b = self.x + (self.width/2), self.y + (self.height/2) #El rectangulo está centrado en (x, y)
        self.a = self.x - (self.width/2), self.y + (self.height/2)              #(a), (b), (c), (d) son los vertices

        self.a = rotation((self.x, self.y), self.a, math.radians(self.angle)) # Apply rotation to get correct corner positions
        self.b = rotation((self.x, self.y), self.b, math.radians(self.angle))
        self.c = rotation((self.x, self.y), self.c, math.radians(self.angle))
        self.d = rotation((self.x, self.y), self.d, math.radians(self.angle))

        self.collision_points = [(self.x, self.y)] * 5
        new_collision_points = []
        angle_offsets = [0, 45, -45, 90, -90]

        for point, angle_offset in zip(self.collision_points, angle_offsets):
            angle = self.angle + angle_offset
            while self.is_track(point):
                point = move_point(point, angle, 10)
            while not self.is_track(point):
                point = move_point(point, angle, -1)
            new_collision_points.append(point)

        self.collision_points = new_collision_points
        
        cx = self.center[0]
        cy = self.center[1]
        self.distances = [int(calculateDistance(self.center[0], self.center[1], cp[0], cp[1])) for cp in self.collision_points]

    def draw(self, display):
        rotated_image = pygame.transform.rotate(self.car_image, -self.angle-180)
        rect_rotated_image = rotated_image.get_rect()
        rect_rotated_image.center = self.x, self.y
        gameDisplay.blit(rotated_image, rect_rotated_image)

        center = self.x, self.y
        if self.showlines: 
            cs = [self.c1, self.c2, self.c3, self.c4, self.c5]
            [pygame.draw.line(gameDisplay, Color_line, (self.x, self.y), c, 2) for c in cs]

    def showLines(self):
        self.showlines = not self.showlines

    def feedforward(self):
        #Return the output of the network
        #self.inp = np.array([[self.d1], [self.d2], [self.d3], [self.d4], [self.d5], [self.velocity]])
        inp_list = [[dist] for dist in self.distances] + [[self.velocity]]
        self.inp = np.array(inp_list)
        for b, w in zip(self.biases, self.weights):
            self.inp = sigmoid(np.dot(w, self.inp) + b)
        self.outp = self.inp

    def collision(self):
        corners = [self.a, self.b, self.c, self.d]
        has_not_collided = all(map(self.is_track, corners))
        #not_collision = self.is_track(self.a) and self.is_track(self.b) and self.is_track(self.c) and self.is_track(self.d)
        return not has_not_collided

    def resetPosition(self):
        self.x = 120
        self.y = 480
        self.angle = 180

    def takeAction(self): 
        if self.outp.item(0) > 0.5: #Accelerate
            self.set_accel(0.2)
        else:
            self.set_accel(0)
        if self.outp.item(1) > 0.5: #Brake
            self.set_accel(-0.2)
        if self.outp.item(2) > 0.5: #Turn right
            self.rotate(-5)
        if self.outp.item(3) > 0.5: #Turn left
            self.rotate(5) 

# Global variables that must come after Car class definition
car = Car(layerSizes) # Car instance driven by player
auxcar = Car(layerSizes) # Used in genome mutation

def displayTexts(): #These is just the text being displayed on pygame window
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
    infotext4 = font.render('Selected: ' + str(selected), True, white)
    infotext5 = font.render('Lines ON', True, white) if lines else font.render('Lines OFF', True, white)
    infotext6 = infotext6 = font.render('Player ON', True, white) if player else font.render('Player OFF', True, white)
    infotext7 = font.render('Mutation: '+ str(mutationRate), True, white)
    infotext8 = font.render('Frames: ' + str(frames), True, white)
    infotext9 = font.render('FPS: 30', True, white)
    infotext1Rect = infotext1.get_rect().move(infotextX, infotextY)
    infotexts = [infotext1, infotext2, infotext3, infotext4, infotext5, infotext6, infotext7, infotext8, infotext9]
    inforects = [text.get_rect().move(infotextX, infotextY + index*infotext1Rect.height) for index, text in enumerate(infotexts)]
    
    for text, rect in zip(texts, rects):
        gameDisplay.blit(text, rect)

    for text, rect in zip(infotexts, inforects):
        gameDisplay.blit(text, rect)

for i in range(num_of_nnCars):
    nnCars.append(Car(layerSizes))
   
def redrawGameWindow(): #Called on every frame   
    global alive  
    global frames
    global img

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
        car.update()
        if car.collision():
            car.resetPosition()
            car.update()
        car.draw(gameDisplay)
    if display_info:
        displayTexts() 
    pygame.display.update() #updates the screen
    #Take a screenshot of every frame
    #pygame.image.save(gameDisplay, 'pygameVideo/screenshot' + str(img) + '.jpeg')
    #img += 1

# ---- User input functions ---- #

def show_lines():
    global lines
    car.showLines()
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

    number_track = 2
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
    if len(selectedCars) != 2:
        return
    global generation
    global alive
    global selected
    global auxcar
    global nnCars
    global num_of_nnCars

    for nncar in nnCars:
        nncar.score = 0
    alive = num_of_nnCars
    generation += 1
    selected = 0
    nnCars.clear() 

    for i in range(num_of_nnCars):
        nnCars.append(Car(layerSizes))

    nnCars[0], nnCars[1] = uniformCrossOverWeights(selectedCars[0], selectedCars[1], nnCars[0], nnCars[1])
    nnCars[0], nnCars[1] = uniformCrossOverBiases(selectedCars[0], selectedCars[1], nnCars[0], nnCars[1])

    for i in range(2, num_of_nnCars-2, 2):
        nnCars[i], nnCars[i+1] = copy_weights(nnCars[0], nnCars[i]), copy_weights(nnCars[1], nnCars[i+1])
        nnCars[i], nnCars[i+1] = copy_biases(nnCars[0], nnCars[i]), copy_weights(nnCars[1], nnCars[i+1])

    nnCars[num_of_nnCars-2] = selectedCars[0]
    nnCars[num_of_nnCars-1] = selectedCars[1]
    
    for car in nnCars[-2:]:
        car.car_image = green_small_car
        car.resetPosition()
        car.collided = False

    for i in range(num_of_nnCars-2):
        for _ in range(mutationRate): # j = number of mutations
            nnCars[i], auxcar = mutateOneWeightGene(nnCars[i], auxcar)
            nnCars[i], auxcar = mutateOneWeightGene(auxcar, nnCars[i])
            nnCars[i], auxcar = mutateOneBiasesGene(nnCars[i], auxcar)
            nnCars[i], auxcar = mutateOneBiasesGene(auxcar, nnCars[i])
    if number_track != 1:
        for nncar in nnCars:
            nncar.x = 140
            nncar.y = 610 

    selectedCars.clear()

def next_gen_and_map():
    global selectedCars
    if len(selectedCars) != 2:
        return
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
    for nncar in nnCars:
        if number_track == 1:
            nncar.x = 120
            nncar.y = 480
        elif number_track == 2:
            nncar.x = 100
            nncar.y = 300

def clear_selected_cars():
    global selected
    selected = 0
    selectedCars.clear()

def mouseclick():
    global selected
    global alive
    global nnCars
    #This returns a tuple:
    #(leftclick, middleclick, rightclick)
    #Each one is a boolean integer representing button up/down.
    mouses = pygame.mouse.get_pressed()

    if mouses[0]:
        pos = pygame.mouse.get_pos()
        point = Point(pos[0], pos[1])
        #Revisar la lista de autos y ver cual estaba ahi
        for nncar in nnCars:  
            polygon = Polygon([nncar.a, nncar.b, nncar.c, nncar.d])
            if (polygon.contains(point)):
                if nncar in selectedCars:
                    selectedCars.remove(nncar)
                    selected -= 1
                    if nncar.car_image == white_big_car:
                        nncar.car_image = white_small_car 
                    if nncar.car_image == green_big_car:
                        nncar.car_image = green_small_car
                    if nncar.collided:
                        nncar.velocity = 0
                        nncar.acceleration = 0
                    nncar.update()
                else:
                    if len(selectedCars) < 2:
                        selectedCars.append(nncar)
                        selected +=1
                        if nncar.car_image == white_small_car:
                            nncar.car_image = white_big_car  
                        if nncar.car_image == green_small_car:
                            nncar.car_image = green_big_car  
                        if nncar.collided:
                            nncar.velocity = 0
                            nncar.acceleration = 0
                        nncar.update()
                break

    if mouses[2]:
        pos = pygame.mouse.get_pos()
        point = Point(pos[0], pos[1])
        for nncar in nnCars:  
            polygon = Polygon([nncar.a, nncar.b, nncar.c, nncar.d])
            if (polygon.contains(point)):
                if nncar not in selectedCars:
                    nnCars.remove(nncar)
                    if not nncar.collided:
                        alive -= 1
                break

def swap_player():
    global player
    player = not player

def swap_display():
    global display_info
    display_info = not display_info

def change_mutation_rate(input):
    global mutationRate
    mutationRate = 10 * (input - ord('0'))

# ---- Player input reaction ---- #

def key_event():
    if event.key == ord('l'):
        show_lines()
    if event.key == ord('c'):
        scrap_stuck_cars()
    if event.key == ord('a'):
        swap_player()
    if event.key == ord('d'):
        swap_display()
    if event.key == ord('n'):
        next_map()
    if event.key == ord('z'):
        clear_selected_cars()
    if event.key == ord('b'):
        next_gen()
    if event.key == ord('m'):
        next_gen_and_map()
    if event.key == ord('r'):
        reload_map()
    if ord('0') <= event.key and event.key <= ord('9'):
        change_mutation_rate(event.key)

# ---- Game loop ---- #

while True:
    for event in pygame.event.get(): #Check for events
        if event.type == pygame.QUIT:
            pygame.quit() #quits
            quit()
            
        if event.type == pygame.KEYDOWN: #If user uses the keyboard
            key_event()

        if event.type == pygame.MOUSEBUTTONDOWN:
            mouseclick()

    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        car.rotate(-5)
    if keys[pygame.K_RIGHT]:
        car.rotate(5)
    if keys[pygame.K_UP]:
        car.set_accel(0.2)
    else:
        car.set_accel(0)
    if keys[pygame.K_DOWN]:
        car.set_accel(-0.2)

    redrawGameWindow()   
    clock.tick(FPS)
