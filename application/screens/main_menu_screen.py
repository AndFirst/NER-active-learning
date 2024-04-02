from kivy.uix.screenmanager import Screen
from kivy.lang import Builder

kv_string = """
<MainMenuScreen>:
    BoxLayout:
        orientation: 'horizontal'
        GridLayout:
            cols: 1
            size_hint_x: None
            pos_hint: {'top':1}
            width: 200
            color: 0, 0, 0, 1
            Button:
                text: 'Adnotuj'
            Button:
                text: 'Zbiór danych'
            Button:
                text: 'Etykiety'
            Button:
                text: 'Metryki'     
"""

Builder.load_string(kv_string)


class MainMenuScreen(Screen):
    pass
