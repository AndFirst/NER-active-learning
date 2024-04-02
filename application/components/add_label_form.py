from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.lang import Builder
from components.label_row import LabelRow
from kivy.clock import Clock
from kivy.properties import ObjectProperty
# Definicja kodu Kivy jako string
kv_string = '''
<AddLabelForm>:
    orientation: 'vertical'
    ScrollView:
        scroll_type: ['bars']
        bar_width: 20
        # canvas.before:
        #     Color:
        #         rgba: 0, 0, 0, 1  # Czarny kolor
        #     Rectangle:
        #         pos: self.pos
        #         size: self.size
        GridLayout:
            padding: 50
            spacing: 10
            cols: 1
            size_hint_y: None
            height: self.minimum_height
            LabelRow:
'''

# Wczytanie kodu Kivy z stringa
Builder.load_string(kv_string)


class AddLabelForm(BoxLayout):
    pass
