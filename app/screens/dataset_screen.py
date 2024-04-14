from kivy.uix.screenmanager import Screen
from kivy.lang import Builder
from kivy.uix.popup import Popup
from kivy.uix.filechooser import FileChooserIconView
from kivy.properties import StringProperty
from kivy.utils import platform
import os
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.button import Button

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
                on_press: root.open_file_chooser()
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
        shared_data = kwargs.pop("shared_data", None)
        super(DatasetScreen, self).__init__(**kwargs)
        self.shared_data = shared_data

        self.ids.prev_next_buttons.on_back = self.go_to_create_project
        self.ids.prev_next_buttons.on_next = self.check_file

    def go_to_create_project(self):
        self.manager.current = "create_project"

    def go_to_add_labels(self):
        self.manager.current = "add_labels"

    def open_file_chooser(self):
        app_path = App.get_running_app().home_dir
        if platform == "win":
            filters = ["*.csv", "*.json", "*.jsonl"]
        else:
            filters = [
                lambda folder, filename: any(
                    filename.endswith(ext)
                    for ext in (".csv", ".json", ".jsonl")
                )
            ]

        file_chooser = FileChooserIconView(
            path=app_path, filters=filters, dirselect=False
        )
        file_chooser.bind(on_submit=self.on_submit)

        choose_button = Button(
            text="Choose dataset", size_hint=(None, None), size=(150, 50)
        )
        choose_button.bind(
            on_press=lambda instance: self.on_submit(
                file_chooser, file_chooser.selection, None
            )
        )
        file_chooser.add_widget(choose_button)

        self.popup = Popup(
            title="Choose file", content=file_chooser, size_hint=(0.9, 0.9)
        )
        self.popup.open()

    def on_submit(self, instance, selection, touch):
        if selection:
            self.selected_file = selection[0]
        self.popup.dismiss()

    def check_file(self):
        if self.selected_file:
            if os.path.exists(self.selected_file):
                self.shared_data.dataset_path = self.selected_file
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
