import pygame
import math
import random
import numpy as np
from pygame.math import Vector3
import json
import os
from datetime import datetime

# Inicialización de Pygame y configuración
pygame.init()
pygame.font.init()

# Constantes globales
WIDTH = 1280
HEIGHT = 720
FPS = 60
VERSION = "1.0.0"

# Colores
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GRAY = (128, 128, 128)

class MathUtils:
    @staticmethod
    def normalize_vector(vector):
        length = math.sqrt(sum(x * x for x in vector))
        return tuple(x / length if length != 0 else 0 for x in vector)

    @staticmethod
    def dot_product(v1, v2):
        return sum(x * y for x, y in zip(v1, v2))

    @staticmethod
    def cross_product(v1, v2):
        return (
            v1[1] * v2[2] - v1[2] * v2[1],
            v1[2] * v2[0] - v1[0] * v2[2],
            v1[0] * v2[1] - v1[1] * v2[0]
        )

    @staticmethod
    def matrix_multiply(matrix, vector):
        return [sum(m * v for m, v in zip(row, vector)) for row in matrix]

class Transform:
    def __init__(self):
        self.position = Vector3(0, 0, 0)
        self.rotation = Vector3(0, 0, 0)
        self.scale = Vector3(1, 1, 1)
        self.matrix = np.identity(4)
        self.update_matrix()

    def update_matrix(self):
        # Matriz de traslación
        translation = np.array([
            [1, 0, 0, self.position.x],
            [0, 1, 0, self.position.y],
            [0, 0, 1, self.position.z],
            [0, 0, 0, 1]
        ])

        # Matrices de rotación
        rx = np.array([
            [1, 0, 0, 0],
            [0, math.cos(self.rotation.x), -math.sin(self.rotation.x), 0],
            [0, math.sin(self.rotation.x), math.cos(self.rotation.x), 0],
            [0, 0, 0, 1]
        ])

        ry = np.array([
            [math.cos(self.rotation.y), 0, math.sin(self.rotation.y), 0],
            [0, 1, 0, 0],
            [-math.sin(self.rotation.y), 0, math.cos(self.rotation.y), 0],
            [0, 0, 0, 1]
        ])

        rz = np.array([
            [math.cos(self.rotation.z), -math.sin(self.rotation.z), 0, 0],
            [math.sin(self.rotation.z), math.cos(self.rotation.z), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # Matriz de escala
        scale = np.array([
            [self.scale.x, 0, 0, 0],
            [0, self.scale.y, 0, 0],
            [0, 0, self.scale.z, 0],
            [0, 0, 0, 1]
        ])

        # Combinar todas las transformaciones
        self.matrix = translation @ rz @ ry @ rx @ scale

class Camera:
    def __init__(self):
        self.transform = Transform()
        self.fov = 60
        self.near = 0.1
        self.far = 1000.0
        self.aspect_ratio = WIDTH / HEIGHT
        self.projection_matrix = self.create_projection_matrix()
        self.view_matrix = np.identity(4)
        self.movement_speed = 5.0
        self.rotation_speed = 2.0
        self.update_matrices()

    def create_projection_matrix(self):
        fov_rad = math.radians(self.fov)
        f = 1.0 / math.tan(fov_rad / 2)
        q = self.far / (self.far - self.near)

        return np.array([
            [f / self.aspect_ratio, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, q, -self.near * q],
            [0, 0, 1, 0]
        ])

    def update_matrices(self):
        # Actualizar matriz de vista
        forward = Vector3(
            math.sin(self.transform.rotation.y) * math.cos(self.transform.rotation.x),
            math.sin(self.transform.rotation.x),
            math.cos(self.transform.rotation.y) * math.cos(self.transform.rotation.x)
        )
        right = Vector3(
            math.cos(self.transform.rotation.y),
            0,
            -math.sin(self.transform.rotation.y)
        )
        up = forward.cross(right)

        view = np.identity(4)
        view[:3, 0] = [right.x, right.y, right.z]
        view[:3, 1] = [up.x, up.y, up.z]
        view[:3, 2] = [-forward.x, -forward.y, -forward.z]
        view[:3, 3] = [-self.transform.position.x, -self.transform.position.y, -self.transform.position.z]

        self.view_matrix = view

    def update(self, dt, keys):
        # Movimiento de la cámara
        move = Vector3(0, 0, 0)
        if keys[pygame.K_w]: move.z += 1
        if keys[pygame.K_s]: move.z -= 1
        if keys[pygame.K_a]: move.x -= 1
        if keys[pygame.K_d]: move.x += 1
        if keys[pygame.K_q]: move.y -= 1
        if keys[pygame.K_e]: move.y += 1

        if move.length() > 0:
            move.normalize_ip()
            move *= self.movement_speed * dt

        # Aplicar movimiento en dirección local
        forward = Vector3(
            math.sin(self.transform.rotation.y),
            0,
            math.cos(self.transform.rotation.y)
        )
        right = Vector3(
            math.cos(self.transform.rotation.y),
            0,
            -math.sin(self.transform.rotation.y)
        )
        up = Vector3(0, 1, 0)

        self.transform.position += forward * move.z
        self.transform.position += right * move.x
        self.transform.position += up * move.y

        self.update_matrices()

class Mesh:
    def __init__(self, vertices, faces, color=WHITE):
        self.vertices = vertices
        self.faces = faces
        self.color = color
        self.transform = Transform()
        self.visible = True

    def update(self, dt):
        self.transform.update_matrix()

    @staticmethod
    def create_cube(size=1.0):
        vertices = [
            Vector3(-size, -size, -size),  # 0
            Vector3(size, -size, -size),   # 1
            Vector3(size, size, -size),    # 2
            Vector3(-size, size, -size),   # 3
            Vector3(-size, -size, size),   # 4
            Vector3(size, -size, size),    # 5
            Vector3(size, size, size),     # 6
            Vector3(-size, size, size),    # 7
        ]

        faces = [
            (0, 1, 2, 3),  # Frente
            (5, 4, 7, 6),  # Atrás
            (4, 0, 3, 7),  # Izquierda
            (1, 5, 6, 2),  # Derecha
            (3, 2, 6, 7),  # Arriba
            (4, 5, 1, 0),  # Abajo
        ]

        return Mesh(vertices, faces)

    @staticmethod
    def create_pyramid(size=1.0):
        vertices = [
            Vector3(-size, -size, -size),  # 0
            Vector3(size, -size, -size),   # 1
            Vector3(size, -size, size),    # 2
            Vector3(-size, -size, size),   # 3
            Vector3(0, size, 0),          # 4 (punta)
        ]

        faces = [
            (0, 1, 4),     # Frente
            (1, 2, 4),     # Derecha
            (2, 3, 4),     # Atrás
            (3, 0, 4),     # Izquierda
            (0, 3, 2, 1),  # Base
        ]

        return Mesh(vertices, faces)

    @staticmethod
    def create_sphere(radius=1.0, segments=16):
        vertices = []
        faces = []
        
        # Generar vértices
        for i in range(segments + 1):
            lat = math.pi * (-0.5 + float(i) / segments)
            for j in range(segments):
                lon = 2 * math.pi * float(j) / segments
                x = math.cos(lon) * math.cos(lat) * radius
                y = math.sin(lat) * radius
                z = math.sin(lon) * math.cos(lat) * radius
                vertices.append(Vector3(x, y, z))

        # Generar caras
        for i in range(segments):
            for j in range(segments):
                first = i * segments + j
                second = first + 1
                third = (i + 1) * segments + j
                fourth = third + 1

                if i != segments - 1:
                    faces.append((first, second, fourth, third))

        return Mesh(vertices, faces)

class Scene:
    def __init__(self):
        self.objects = []
        self.camera = Camera()
        self.selected_object = None

    def add_object(self, obj):
        self.objects.append(obj)

    def remove_object(self, obj):
        if obj in self.objects:
            self.objects.remove(obj)

    def update(self, dt, keys):
        self.camera.update(dt, keys)
        for obj in self.objects:
            obj.update(dt)

    def render(self, screen):
        # Ordenar objetos por distancia a la cámara (painter's algorithm)
        sorted_objects = sorted(
            self.objects,
            key=lambda obj: (obj.transform.position - self.camera.transform.position).length(),
            reverse=True
        )

        for obj in sorted_objects:
            if not obj.visible:
                continue

            # Transformar y proyectar vértices
            transformed_vertices = []
            for vertex in obj.vertices:
                # Transformar al espacio mundial
                world_pos = np.array([vertex.x, vertex.y, vertex.z, 1.0])
                world_pos = obj.transform.matrix @ world_pos

                # Transformar al espacio de la cámara
                view_pos = self.camera.view_matrix @ world_pos

                # Proyección
                if view_pos[2] > self.camera.near:
                    proj_pos = self.camera.projection_matrix @ view_pos
                    if proj_pos[3] != 0:
                        proj_pos = proj_pos / proj_pos[3]
                        screen_x = (proj_pos[0] + 1) * WIDTH / 2
                        screen_y = (-proj_pos[1] + 1) * HEIGHT / 2
                        transformed_vertices.append((screen_x, screen_y))
                    else:
                        transformed_vertices.append(None)
                else:
                    transformed_vertices.append(None)

            # Dibujar caras
            for face in obj.faces:
                points = []
                skip_face = False
                for vertex_idx in face:
                    if transformed_vertices[vertex_idx] is None:
                        skip_face = True
                        break
                    points.append(transformed_vertices[vertex_idx])

                if not skip_face and len(points) >= 3:
                    pygame.draw.polygon(screen, obj.color, points, 1)

class UI:
    def __init__(self, scene):
        self.scene = scene
        self.font = pygame.font.Font(None, 24)
        self.buttons = []
        self.setup_ui()

    def setup_ui(self):
        button_width = 100
        button_height = 30
        margin = 10
        y = margin

        # Botones para crear objetos
        self.add_button("Cubo", (margin, y, button_width, button_height), self.add_cube)
        y += button_height + margin
        self.add_button("Pirámide", (margin, y, button_width, button_height), self.add_pyramid)
        y += button_height + margin
        self.add_button("Esfera", (margin, y, button_width, button_height), self.add_sphere)
        y += button_height + margin
        self.add_button("Borrar", (margin, y, button_width, button_height), self.delete_selected)

    def add_button(self, text, rect, callback):
        self.buttons.append({
            'text': text,
            'rect': pygame.Rect(rect),
            'callback': callback
        })

    def add_cube(self):
        cube = Mesh.create_cube()
        cube.transform.position = Vector3(
            random.uniform(-5, 5),
            random.uniform(-5, 5),
            random.uniform(-5, 5)
        )
        self.scene.add_object(cube)

    def add_pyramid(self):
        pyramid = Mesh.create_pyramid()
        pyramid.transform.position = Vector3(
            random.uniform(-5, 5),
            random.uniform(-5, 5),
            random.uniform(-5, 5)
        )
        self.scene.add_object(pyramid)

    def add_sphere(self):
        sphere = Mesh.create_sphere()
        sphere.transform.position = Vector3(
            random.uniform(-5, 5),
            random.uniform(-5, 5),
            random.uniform(-5, 5)
        )
        self.scene.add_object(sphere)

    def delete_selected(self):
        if self.scene.selected_object:
            self.scene.remove_object(self.scene.selected_object)
            self.scene.selected_object = None

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = pygame.mouse.get_pos()
            for button in self.buttons:
                if button['rect'].collidepoint(mouse_pos):
                    button['callback']()

    def render(self, screen):
        # Dibujar botones
        for button in self.buttons:
            pygame.draw.rect(screen, WHITE, button['rect'], 2)
            text = self.font.render(button['text'], True, WHITE)
            text_rect = text.get_rect(center=button['rect'].center)
            screen.blit(text, text_rect)

        # Mostrar información de la cámara
        camera_info = f"Cámara: {self.scene.camera.transform.position}"
        text = self.font.render(camera_info, True, WHITE)
        screen.blit(text, (10, HEIGHT - 30))

class Application:
    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption(f"Editor 3D - v{VERSION}")
        self.clock = pygame.time.Clock()
        self.running = True
        self.scene = Scene()
        self.ui = UI(self.scene)
        self.mouse_buttons = {1: False, 2: False, 3: False}
        self.last_mouse_pos = (0, 0)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.mouse_buttons[event.button] = True
                self.last_mouse_pos = event.pos
                self.ui.handle_event(event)
            elif event.type == pygame.MOUSEBUTTONUP:
                self.mouse_buttons[event.button] = False
            elif event.type == pygame.MOUSEMOTION:
                if self.mouse_buttons[1]:  # Botón izquierdo
                    dx, dy = event.pos[0] - self.last_mouse_pos[0], event.pos[1] - self.last_mouse_pos[1]
                    self.scene.camera.transform.rotation.y += dx * 0.01
                    self.scene.camera.transform.rotation.x += dy * 0.01
                self.last_mouse_pos = event.pos

    def update(self):
        dt = self.clock.tick(FPS) / 1000.0
        keys = pygame.key.get_pressed()
        self.scene.update(dt, keys)

    def render(self):
        self.screen.fill(BLACK)
        self.scene.render(self.screen)
        self.ui.render(self.screen)
        pygame.display.flip()

    def run(self):
        while self.running:
            self.handle_events()
            self.update()
            self.render()

        pygame.quit()

def main():
    app = Application()
    app.run()

if __name__ == "__main__":
    main()

