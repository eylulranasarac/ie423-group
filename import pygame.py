import pygame
import sys
import random

# Oyun ekranı boyutları
WIDTH, HEIGHT = 400, 600

# Renkler
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Topların boyutu ve hızı
BALL_RADIUS = 30
BALL_SPEED = 5

# Top listesi
balls = []

# Pygame başlat
pygame.init()

# Oyun ekranını oluştur
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Tap to Blast")

# Oyun döngüsü
clock = pygame.time.Clock()

def create_ball():
    x = random.randint(BALL_RADIUS, WIDTH - BALL_RADIUS)
    y = random.randint(0, HEIGHT // 2)
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    return {"x": x, "y": y, "color": color}

def draw_balls():
    for ball in balls:
        pygame.draw.circle(screen, ball["color"], (ball["x"], ball["y"]), BALL_RADIUS)

def move_balls():
    for ball in balls:
        ball["y"] += BALL_SPEED

def main():
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Fare tıklaması algılandığında, topu patlat (liste içinden kaldır)
                for ball in balls:
                    distance = pygame.math.Vector2(event.pos[0] - ball["x"], event.pos[1] - ball["y"]).length()
                    if distance < BALL_RADIUS:
                        balls.remove(ball)

        # Ekranı temizle
        screen.fill(WHITE)

        # Topları oluştur ve hareket ettir
        if random.random() < 0.02:
            balls.append(create_ball())

        move_balls()
        draw_balls()

        # Ekranı güncelle
        pygame.display.flip()

        # FPS sınırlama
        clock.tick(60)

if __name__ == "__main__":
    main()
