from kivy.uix.screenmanager import Screen
from kivy.lang import Builder
from kivy.uix.popup import Popup
from kivy.properties import StringProperty
import os
from plyer import filechooser
from kivy.uix.label import Label
from app.data_types import ProjectFormState

kv_string = """
<DatasetScreen>:
    BoxLayout:
        orientation: 'vertical'
        size_hint: (1, 1)
        padding: 20
        spacing: 10
        BoxLayout:
            canvas.before:
                Color:
                    rgba: 0, 0, 0, 1
                Line:
                    rectangle: (self.x, self.y, self.width, self.height)
                    width: 1
            orientation: 'vertical'
            size_hint_y: 0.8
            Button:
                text: 'Choose file'
                size_hint: None, None
                size: 150, 50
                on_release: root.open_dataset_filechooser()
            Label:
                text: 'Chosen file: ' + root.selected_file
                color: 0, 0, 0, 1
        PrevNextButtons:
            id: prev_next_buttons
            size_hint_y: 0.2
"""

Builder.load_string(kv_string)


class DatasetScreen(Screen):
    selected_file = StringProperty("")
    popup = None

    def __init__(self, **kwargs):
        form_state = kwargs.pop("form_state", None)
        super(DatasetScreen, self).__init__(**kwargs)
        self.form_state: ProjectFormState = form_state
        self.ids.prev_next_buttons.on_back = self.go_to_create_project
        self.ids.prev_next_buttons.on_next = self.check_file

    def go_to_create_project(self):
        self.manager.current = "create_project"

    def go_to_add_labels(self):
        self.manager.current = "add_labels"

    def open_dataset_filechooser(self):
        file_path = filechooser.open_file(
            filters=["*.csv", "*.json", "*.jsonl"],
            title="Select Dataset File",
            multiple=False,
        )
        if file_path:
            self.selected_file = file_path[0]
            self.form_state.dataset_path = self.selected_file

    def check_file(self):
        if self.selected_file:
            if os.path.exists(self.selected_file):
                self.form_state.dataset_path = self.selected_file
                self.manager.current = "add_labels"
            else:
                Popup(
                    title="Error",
                    content=Label(text="Chosen file not exist."),
                    size_hint=(None, None),
                    size=(300, 200),
                ).open()
        else:
            Popup(
                title="Error",
                content=Label(text="File not chosen."),
                size_hint=(None, None),
                size=(300, 200),
            ).open()
