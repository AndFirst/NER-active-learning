from kivy.uix.screenmanager import Screen
from kivy.lang import Builder

kv_string = f"""
<WelcomeScreen>:
    BoxLayout:
        size_hint: 0.5, 0.5
        pos_hint: {{'center_x': 0.5, 'center_y': 0.5}}
        orientation: 'vertical'
        padding: 20
        spacing: 20
        Button:
            text: 'New project'
            size_hint: 1, 1
            on_press: app.root.current = 'create_project'
        Button:
            text: 'Use existing project'
            size_hint: 1, 1
            on_press: app.root.current = 'existing_project'
"""

Builder.load_string(kv_string)


class WelcomeScreen(Screen):
    pass
