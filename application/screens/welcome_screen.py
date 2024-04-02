from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.lang import Builder

from ui_colors import BACKGROUND_COLOR

# Definicja KV
kv_string = f"""
<CenteredContainer>:
    orientation: 'vertical'
    padding: 20
    spacing: 20
    Button:
        text: 'Utwórz nowy projekt'
        size_hint: 1, 1
        on_press: app.root.current = 'create_project'

    Button:
        text: 'Użyj istniejącego projektu'
        size_hint: 1, 1
        on_press: app.root.current = 'existing_project'

<WelcomeScreen>:
    FloatLayout:
        # canvas.before:
        #     Color:
        #         rgba: {BACKGROUND_COLOR} # Kolor tła
        #     Rectangle:
        #         size: self.size
        #         pos: self.pos
        CenteredContainer:
            size_hint: 0.5, 0.5
            pos_hint: {{'center_x': 0.5, 'center_y': 0.5}}
"""

# Załadowanie KV
Builder.load_string(kv_string)


class CenteredContainer(BoxLayout):
    pass


class WelcomeScreen(Screen):
    pass
