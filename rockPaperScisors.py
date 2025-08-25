import pygame
import random

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 600
FPS = 60
NUM_PARTICLES = 50
PARTICLE_RADIUS = 10
ROCK, PAPER, SCISSORS = 0, 1, 2
PARTICLE_TYPES = [ROCK, PAPER, SCISSORS]
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Red, Green, Blue

# Particle class
class Particle:
    def __init__(self, x, y, particle_type):
        self.x = x
        self.y = y
        self.type = particle_type

    def collide(self, other_particle):
        # Collision rules
        if self.type == ROCK and other_particle.type == SCISSORS:
            self.type = ROCK
        elif self.type == PAPER and other_particle.type == ROCK:
            self.type = PAPER
        elif self.type == SCISSORS and other_particle.type == PAPER:
            self.type = SCISSORS

# Create particles
particles = [Particle(random.randint(PARTICLE_RADIUS, WIDTH - PARTICLE_RADIUS),
                      random.randint(PARTICLE_RADIUS, HEIGHT - PARTICLE_RADIUS),
                      random.choice(PARTICLE_TYPES))
             for _ in range(NUM_PARTICLES)]

# Set up the Pygame screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Brownian Particles Simulation")
clock = pygame.time.Clock()

# Main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Update particle positions
    for particle in particles:
        # Implement Brownian motion
        particle.x += random.randint(-1, 1)
        particle.y += random.randint(-1, 1)

        # Ensure particles stay within the screen bounds
        particle.x = max(PARTICLE_RADIUS, min(particle.x, WIDTH - PARTICLE_RADIUS))
        particle.y = max(PARTICLE_RADIUS, min(particle.y, HEIGHT - PARTICLE_RADIUS))

        # Check for collisions with other particles
        for other_particle in particles:
            if particle != other_particle:
                distance = ((particle.x - other_particle.x) ** 2 + (particle.y - other_particle.y) ** 2) ** 0.5
                if distance < 2 * PARTICLE_RADIUS:
                    particle.collide(other_particle)

    # Draw particles
    screen.fill((255, 255, 255))
    for particle in particles:
        pygame.draw.circle(screen, COLORS[particle.type], (int(particle.x), int(particle.y)), PARTICLE_RADIUS)

    # Update the display
    pygame.display.flip()

    # Control the game speed
    clock.tick(FPS)

# Quit Pygame
pygame.quit()