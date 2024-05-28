from kivy.uix.screenmanager import Screen
from kivy.lang import Builder
from kivy.uix.popup import Popup
from kivy.properties import StringProperty
from plyer import filechooser
import os
from kivy.uix.label import Label

from app.project import Project

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
                on_release: root.open_filechooser()
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

    def open_filechooser(self):
        selected_path = filechooser.choose_dir(title="Select Project Folder")
        if selected_path:
            self.selected_path = selected_path[0]

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
                try:
                    project = Project.load(self.selected_path)
                except Exception as e:
                    popup = Popup(
                        title="Error",
                        content=Label(
                            text=str(e),
                            text_size=(
                                360,
                                None,
                            ),
                            halign="left",
                            valign="top",
                        ),
                        size_hint=(None, None),
                        size=(400, 400),
                    )
                    popup.open()
                    return
                self.manager.get_screen("main_menu").project = project
                self.manager.get_screen("main_menu").save_path = (
                    self.selected_path
                )
                self.manager.current = "main_menu"
        else:
            Popup(
                title="Error",
                content=Label(text="Path not chosen."),
                size_hint=(None, None),
                size=(300, 200),
            ).open()
