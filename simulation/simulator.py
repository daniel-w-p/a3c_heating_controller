import pygame

import setup


class Simulator:
    """Simulate building floor with 4 rooms"""
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GREEN = (20, 225, 20)
    RED = (225, 20, 20)
    GREY = (196, 196, 196)
    room_width, room_height = 300, 250
    texts_inside = ["Desired temp: ", "Indoor temp: ", "Floor temp: "]
    texts_outside = ["Outside temperature: "]

    def __init__(self, screen, font):
        self.screen = screen
        self.font = font

    def draw_led(self, x, y, status):
        color = self.RED if status else self.GREY
        pygame.draw.circle(self.screen, color, (x, y), 10)

    def draw_room(self, x, y, width, height, values):
        pygame.draw.rect(self.screen, self.BLACK, [x, y, width, height], 2)
        text_height = y + 10
        self.draw_led(x + width - 15, text_height + 5, values[3])
        for val, desc in zip(values, self.texts_inside):
            if isinstance(val, float):
                val = round(val, 4)
            img = self.font.render(desc + str(val), True, self.BLACK)
            self.screen.blit(img, (x + 10, text_height))
            text_height += img.get_height() + 10

    def draw_outside(self, width, height, values):
        x, y = 0, 0
        text_height = y + 10
        pygame.draw.rect(self.screen, self.BLACK, [x, y, width, height], 2)
        for val, desc in zip(values, self.texts_outside):
            if isinstance(val, float):
                val = round(val, 4)
            img = self.font.render(desc + str(val), True, self.BLACK)
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

            # rooms
            start_x, start_y = 200, 200
            self.draw_room(start_x, start_y, self.room_width, self.room_height, values[0])
            self.draw_room(start_x + self.room_width, start_y, self.room_width, self.room_height, values[1])
            self.draw_room(start_x, start_y + self.room_height, self.room_width, self.room_height, values[2])
            self.draw_room(start_x + self.room_width, start_y + self.room_height, self.room_width, self.room_height, values[3])

            # outside
            self.draw_outside(self.room_width, 100, values[4])

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            clock.tick(setup.gui['FRAME_RATE'])

            pygame.display.update()

