from kivy.uix.boxlayout import BoxLayout
from kivy.properties import StringProperty
from kivy.lang import Builder

kv_string = """
<PrevNextButtons>:
    BoxLayout:
        canvas:
            Color:
                rgba: 1, 1, 1, 1  # Czarny kolor
            Rectangle:
                pos: self.pos
                size: self.size
        size_hint_y: None
        maximum_height: 100 
        height: 100
        padding: 10
        spacing: 10
        Button:
            id: back_button
            text: root.back_text
            on_press: root.on_back()
        Button:
            id: next_button
            text: root.next_text
            on_press: root.on_next()
"""

Builder.load_string(kv_string)


class PrevNextButtons(BoxLayout):
    back_text = StringProperty()
    next_text = StringProperty()
    on_back = None
    on_next = None
