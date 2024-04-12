import pygame


class Simulator:
    """Simulate building floor with 4 rooms"""
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    room_width, room_height = 200, 150
    texts_inside = ["Text 1", "Text 2", "Text 3"]
    texts_outside = ["Outside 1", "Outside 2", "Outside 3"]

    def __init__(self, screen, font):
        self.screen = screen
        self.font = font

    def draw_room(self, x, y, width, height, texts):
        pygame.draw.rect(self.screen, self.BLACK, [x, y, width, height], 2)
        text_height = y + 10
        for text in texts:
            img = self.font.render(text, True, self.BLACK)
            self.screen.blit(img, (x + 10, text_height))
            text_height += img.get_height() + 10

    def run(self, start_time, model, env, callback):
        running = True
        state = env.reset()

        while running:
            texts = env.get_values_as_strings()

            state = callback(model, env, state, start_time)

            self.screen.fill(self.WHITE)

            # rooms
            self.draw_room(100, 100, self.room_width, self.room_height, texts[0])
            self.draw_room(310, 100, self.room_width, self.room_height, texts[1])
            self.draw_room(100, 260, self.room_width, self.room_height, texts[2])
            self.draw_room(310, 260, self.room_width, self.room_height, texts[3])

            # outside
            self.draw_room(550, 100, self.room_width, self.room_height, texts[4])

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            pygame.display.flip()

