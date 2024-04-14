from kivy.uix.button import Button
from kivy.uix.screenmanager import Screen
from kivy.lang import Builder
from kivy.uix.popup import Popup
from kivy.uix.filechooser import FileChooserIconView
from kivy.properties import StringProperty
from kivy.utils import platform
import os
from kivy.app import App
from kivy.uix.label import Label

kv_string = """
<ExistingProjectScreen>:
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
                text: 'Choose directory'
                size_hint: None, None
                size: 150, 50
                on_press: root.open_file_chooser()
            Label:
                text: 'Path: ' + root.selected_path
                color: 0, 0, 0, 1  # Kolor tekstu: czarny
        PrevNextButtons:
            size_hint_y: 0.2
            id: prev_next_buttons
"""

Builder.load_string(kv_string)


class ExistingProjectScreen(Screen):
    selected_path = StringProperty("")
    popup = None

    def __init__(self, **kwargs):
        super(ExistingProjectScreen, self).__init__(**kwargs)
        self.ids.prev_next_buttons.on_back = self.go_to_welcome
        self.ids.prev_next_buttons.on_next = self.check_files

    def go_to_welcome(self):
        self.manager.current = "welcome"

    def go_to_main_menu(self):
        self.manager.current = "main_menu"

    def open_file_chooser(self):
        app_path = App.get_running_app().home_dir
        if platform == "win":
            filters = ["*"]
        else:
            filters = [
                lambda folder, filename: os.path.isdir(
                    os.path.join(folder, filename)
                )
            ]

        file_chooser = FileChooserIconView(
            path=app_path, filters=filters, dirselect=True
        )
        file_chooser.bind(on_submit=self.on_submit)

        choose_button = Button(
            text="Choose", size_hint=(None, None), size=(150, 50)
        )
        choose_button.bind(
            on_press=lambda instance: self.on_submit(
                file_chooser, file_chooser.selection, None
            )
        )
        file_chooser.add_widget(choose_button)

        self.popup = Popup(
            title="Choose directory",
            content=file_chooser,
            size_hint=(0.9, 0.9),
        )
        self.popup.open()

    def on_submit(self, instance, selection, touch):
        if selection:
            self.selected_path = selection[0]
        self.popup.dismiss()

    def check_files(self):
        if self.selected_path:
            setup_path = "project.json"

            dataset_files = {
                "unlabeled.json",
                "unlabeled.jsonl",
                "unlabeled.csv",
            }

            existing_files = [
                f
                for f in dataset_files
                if os.path.exists(os.path.join(self.selected_path, f))
            ]

            if not os.path.exists(
                os.path.join(self.selected_path, setup_path)
            ):
                Popup(
                    title="Error",
                    content=Label(
                        text="Data inconsistency: not found project setup file."
                    ),
                    size_hint=(None, None),
                    size=(400, 200),
                ).open()

            elif len(existing_files) != 1:
                Popup(
                    title="Error",
                    content=Label(
                        text="Data inconsistency: \nThere must be exactly one file with unlabeled data."
                    ),
                    size_hint=(None, None),
                    size=(400, 200),
                ).open()
            else:
                self.manager.current = "main_menu"
        else:
            Popup(
                title="Error",
                content=Label(text="Path not chosen."),
                size_hint=(None, None),
                size=(300, 200),
            ).open()
