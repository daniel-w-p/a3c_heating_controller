import pygame

from setup import gui


class Simulator:
    """Simulate building floor with 4 rooms"""
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GREEN = (20, 225, 20)
    RED = (225, 20, 20)
    GRAY = (196, 196, 196)
    LIGHT_GREEN = (140, 255, 140)
    DARK_GREEN = (20, 120, 20)
    room_width, room_height = 300, 250
    texts_inside = ["Desired temp: ", "Indoor temp: ", "Floor temp: "]
    texts_outside = ["Outside temperature: "]

    def __init__(self, screen, font):
        self.screen = screen
        self.font = font
        self.frame_rate = gui['FRAME_RATE']
        self.buttons = [
            {"rect": pygame.Rect(gui['SCREEN_WIDTH'] - 180, 40, 35, 35), "text": "1x", "speed": 1},
            {"rect": pygame.Rect(gui['SCREEN_WIDTH'] - 140, 40, 35, 35), "text": "10x", "speed": 10},
            {"rect": pygame.Rect(gui['SCREEN_WIDTH'] - 100, 40, 35, 35), "text": "20x", "speed": 20}
        ]

    def draw_buttons(self):
        img = self.font.render("For '1x' one second (real) is one minute for environment", True, self.BLACK)
        self.screen.blit(img, (gui['SCREEN_WIDTH'] // 2, 10))
        for button in self.buttons:
            if button["speed"] == self.frame_rate:
                pygame.draw.rect(self.screen, self.LIGHT_GREEN, button["rect"])
                pygame.draw.rect(self.screen, self.DARK_GREEN, button["rect"], 2)  # Frame
            else:
                pygame.draw.rect(self.screen, self.GRAY, button["rect"])
                pygame.draw.rect(self.screen, self.BLACK, button["rect"], 2)  # Frame

            text_surface = self.font.render(button["text"], True, self.BLACK)
            text_rect = text_surface.get_rect(center=button["rect"].center)
            self.screen.blit(text_surface, text_rect)

    def draw_led(self, x, y, status):
        color = self.RED if status else self.GRAY
        pygame.draw.circle(self.screen, color, (x, y), 10)

    def draw_screen_legend(self):
        x = gui['SCREEN_WIDTH'] - 160
        y = gui['SCREEN_HEIGHT'] - 200
        img = self.font.render("LEGEND", True, self.BLACK)
        self.screen.blit(img, (x - 20, y - 30))
        self.draw_led(x, y + 12, True)
        img = self.font.render("Heating ON", True, self.BLACK)
        self.screen.blit(img, (x + 12, y))
        y += 30
        self.draw_led(x, y + 12, False)
        img = self.font.render("Heating OFF", True, self.BLACK)
        self.screen.blit(img, (x + 12, y))

    def draw_room(self, x, y, width, height, values):
        pygame.draw.rect(self.screen, self.BLACK, [x, y, width, height], 2)
        text_height = y + 10
        self.draw_led(x + width - 15, text_height + 5, values[3])
        for val, desc in zip(values, self.texts_inside):
            if isinstance(val, float):
                val = round(val, 2)
            img = self.font.render(desc + str(val) + " °C", True, self.BLACK)
            self.screen.blit(img, (x + 10, text_height))
            text_height += img.get_height() + 10

    def draw_outside(self, width, height, values):
        x, y = 0, 0
        text_height = y + 10
        pygame.draw.rect(self.screen, self.BLACK, [x, y, width, height], 2)
        for val, desc in zip(values, self.texts_outside):
            if isinstance(val, float):
                val = round(val, 2)
            img = self.font.render(desc + str(val) + " °C", True, self.BLACK)
            self.screen.blit(img, (x + 10, text_height))
            text_height += img.get_height() + 10

        val = values[1]
        desc = "Time (dd:hh:mm): "
        val = "{:02} : {:02} : {:02}".format(val // 1440, (val % 1440) // 60, val % 60)
        img = self.font.render(desc + str(val), True, self.BLACK)
        self.screen.blit(img, (x + 10, text_height))
        text_height += img.get_height() + 10

    def run(self, model, env, callback):
        running = True
        clock = pygame.time.Clock()

        state = env.reset()

        while running:
            values = env.get_values()

            state = callback(model, env, state)

            self.screen.fill(self.WHITE)

            self.draw_buttons()
            self.draw_screen_legend()

            # rooms
            start_x, start_y = 150, 200
            self.draw_room(start_x, start_y, self.room_width, self.room_height, values[0])
            self.draw_room(start_x + self.room_width, start_y, self.room_width, self.room_height, values[1])
            self.draw_room(start_x, start_y + self.room_height, self.room_width, self.room_height, values[2])
            self.draw_room(start_x + self.room_width, start_y + self.room_height, self.room_width, self.room_height,
                           values[3])

            # outside
            self.draw_outside(self.room_width, 100, values[4])

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # LPM
                        for button in self.buttons:
                            if button["rect"].collidepoint(event.pos):
                                self.frame_rate = button["speed"]
                                print(f"Actual frame rate: {self.frame_rate}")

            clock.tick(self.frame_rate)

            pygame.display.update()
