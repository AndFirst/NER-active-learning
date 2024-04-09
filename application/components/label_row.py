from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput

kv_string = """
<LabelRow>:
    orientation: 'horizontal'
    size_hint_y: None
    height: 60
    spacing: 20
    TextInput:
        id: text_input
        multiline: False
        size_hint_x: 0.8 
        on_text_validate: root.add_new_row() if self.text else None
    Button:
        text: "Usuń"
        size_hint_x: 0.2 
        height: root.height 
        on_release: root.remove_row()
"""

Builder.load_string(kv_string)


class LabelRow(BoxLayout):
    def remove_row(self):
        parent = self.parent
        if parent and len(parent.children) > 1:
            parent.remove_widget(self)

    def add_new_row(self):
        parent = self.parent
        if parent:
            # Sprawdź, czy istnieje pusty wiersz
            empty_row_exists = any(
                child.ids.text_input.text == ""
                for child in parent.children
                if isinstance(child, LabelRow)
            )
            if empty_row_exists:
                # Przenieś fokus na pusty wiersz
                for child in parent.children:
                    if isinstance(child, LabelRow) and child.ids.text_input.text == "":
                        child.ids.text_input.focus = True
                        break
            else:
                # Utwórz nowy wiersz i przenieś fokus na niego
                new_row = LabelRow()
                parent.add_widget(new_row)
                new_row.ids.text_input.focus = True
