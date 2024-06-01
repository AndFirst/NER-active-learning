import random
import string
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.colorpicker import ColorWheel
from kivy.uix.popup import Popup
from kivy.properties import ListProperty


class LabelTextInput(TextInput):
    max_length = 10
    allowed_characters = set(string.ascii_letters + string.digits + "-_")

    def insert_text(self, substring, from_undo=False):
        if all(c in self.allowed_characters for c in substring):
            if len(self.text) + len(substring) <= self.max_length:
                return super().insert_text(substring, from_undo=from_undo)
        return False


Builder.load_string(
    """
<LabelRow>:
    orientation: 'horizontal'
    size_hint_y: None
    height: 35
    spacing: 20

    LabelTextInput:
        id: text_input
        font_size: 20
        multiline: False
        size_hint_x: 0.6
        hint_text: "Label e.g. GEO_123 (max 10 chars)"
        on_text_validate: root.add_new_row() if self.text else None

    Button:
        id: color_display
        text: ""
        size_hint_x: 0.2
        height: root.height
        background_color: root.color
        background_normal: ''
        background_down: ''
        on_release: root.show_color_picker()

    Button:
        text: "Remove"
        size_hint_x: 0.2
        height: root.height
        on_release: root.remove_row()
"""
)


class LabelRow(BoxLayout):
    color = ListProperty([0.5, 0.5, 0.5, 1])
    max_length = 20

    def __init__(self, **kwargs):
        super(LabelRow, self).__init__(**kwargs)
        self.popup = None
        self.color = [random.random() for _ in range(3)] + [1]

    @property
    def label(self):
        return self.ids.text_input.text

    def remove_row(self):
        parent = self.parent
        if parent and len(parent.children) > 1:
            parent.remove_widget(self)

    def add_new_row(self):
        parent = self.parent
        if parent:
            existing_labels = [
                child.ids.text_input.text.strip().lower()
                for child in parent.children
                if isinstance(child, LabelRow) and child is not self
            ]
            if len(existing_labels) >= 11:
                content = Label(text="Label limit reached (12 labels).", halign="center")
                popup = Popup(
                    title="Error",
                    content=content,
                    size_hint=(None, None),
                    size=(400, 200),
                )
                popup.open()
                return

            new_label_text = self.ids.text_input.text.strip().lower()
            if new_label_text in existing_labels:
                content = Label(text="This label already exists.", halign="center")
                popup = Popup(
                    title="Error",
                    content=content,
                    size_hint=(None, None),
                    size=(400, 200),
                )
                popup.open()
                return

            empty_row_exists = any(child.ids.text_input.text == "" for child in parent.children if isinstance(child, LabelRow))
            if empty_row_exists:
                for child in parent.children:
                    if isinstance(child, LabelRow) and child.ids.text_input.text == "":
                        child.ids.text_input.focus = True
                        break
            else:
                new_row = LabelRow()
                self.parent.add_widget(new_row)
                new_row.ids.text_input.focus = True

    def show_color_picker(self):
        color_picker = ColorWheel(color=self.color)
        color_picker.bind(color=self.on_color_select)
        self.popup = Popup(
            title="Choose color",
            content=color_picker,
            size_hint=(None, None),
            size=(400, 400),
        )
        self.popup.open()

    def on_color_select(self, instance, color):
        self.color = color
        self.ids.color_display.background_color = color
        self.popup.dismiss()
