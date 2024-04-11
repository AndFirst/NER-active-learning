from kivy.uix.boxlayout import BoxLayout
from kivy.lang import Builder

kv_string = """
<PrevNextButtons>:
    orientation: 'horizontal'
    padding: 10
    spacing: 10
    canvas.before:
        Color:
            rgba: 0, 0, 0, 1
        Line:
            rectangle: (self.x, self.y, self.width, self.height)
            width: 1
    Button:
        id: back_button
        text: "Back"
        on_press: root.on_back()
    Button:
        id: next_button
        text: "Next"
        on_press: root.on_next()
"""

Builder.load_string(kv_string)


class PrevNextButtons(BoxLayout):
    on_back = None
    on_next = None
