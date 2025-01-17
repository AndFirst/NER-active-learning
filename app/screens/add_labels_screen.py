from typing import List

from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.screenmanager import Screen
from kivy.lang import Builder

from app.data_types import LabelData

kv_string = """
<AddLabelsScreen>:
    BoxLayout:
        orientation: 'vertical'
        size_hint: (1, 1)
        padding: 20
        spacing: 10
        AddLabelForm:
            id: add_label_form
            size_hint_y: 0.8
        PrevNextButtons:
            id: prev_next_buttons
            size_hint_y: 0.2
"""

Builder.load_string(kv_string)


class AddLabelsScreen(Screen):
    def __init__(self, **kwargs):
        form_state = kwargs.pop("form_state", None)
        super(AddLabelsScreen, self).__init__(**kwargs)
        self.form_state = form_state
        self.ids.prev_next_buttons.on_back = self.go_to_data_set
        self.ids.prev_next_buttons.on_next = self.go_to_summary

    def go_to_data_set(self):
        self.manager.current = "data_set"

    def go_to_summary(self):
        labels = [LabelData(label=item.label, color=item.color) for item in self.ids.add_label_form.label_rows]
        if not self.validate_labels(labels):
            return
        self.form_state.labels = labels
        self.manager.current = "summary"

    def validate_labels(self, labels: List[LabelData]) -> bool:
        if any(label.is_empty() for label in labels):
            self.show_error_popup("Labels cannot be empty")
            return False
        elif len(set(labels)) != len(labels):
            self.show_error_popup("Labels must be unique")
            return False
        return True

    def show_error_popup(self, message: str):
        content = Label(text=message, halign="center")
        popup = Popup(
            title="Error",
            content=content,
            size_hint=(None, None),
            size=(400, 200),
        )
        popup.open()
