from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout

kv_string = """
<LabelRow>:
    orientation: 'horizontal'
    size_hint_y: None
    height: 60
    spacing: 20
    TextInput:
        multiline: False
        size_hint_x: 0.8  # Ustawienie szerokości TextInput na 80% szerokości LabelRow
    Button:
        text: "Usuń"
        size_hint_x: 0.2  # Ustawienie szerokości Button na 20% szerokości LabelRow
        height: root.height  # Ustawienie wysokości Button na wysokość LabelRow
"""

Builder.load_string(kv_string)


class LabelRow(BoxLayout):
    pass

    # def __init__(self, on_add_row, on_delete_row, text, **kwargs):
    #     super(LabelRow, self).__init__(**kwargs)
    #     self.orientation = 'horizontal'
    #     self.size_hint_y = None
    #     self.height = 60

    #     # Pole tekstowe
    #     self.text_input = LabelInput(
    #         on_add_row=on_add_row,
    #         text=text,
    #         readonly=False,
    #         size_hint=(0.5, None),
    #         height=40,
    #         multiline=False
    #     )
    #     self.add_widget(self.text_input)

    #     # Przycisk "Usuń"
    #     self.delete_button = Button(
    #         text="Usuń", size_hint=(0.2, None), height=40)
    #     self.delete_button.bind(
    #         on_release=lambda instance: on_delete_row(self))
    #     self.add_widget(self.delete_button)
