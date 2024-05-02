from kivy.uix.screenmanager import Screen
from kivy.lang import Builder
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.dropdown import DropDown

from data_types import ProjectData

from file_operations import create_unique_folder_name

kv_string = """
<CreateProjectScreen>:
    BoxLayout:
        orientation: 'vertical'
        Label:
            color: 0, 0, 0, 1
            text: 'Project info:'
            size_hint: (1, 0.3)
            font_size: '30sp'
            bold: True
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
            padding: (100, 10)
            spacing: 20
            TextInput:
                id: name_input
                hint_text: 'Project name'
                multiline: False
            TextInput:
                id: description_input
                hint_text: 'Description'
            Button:
                id: model_button
                text: 'Choose a model'
                on_release: root.model_dropdown.open(self)
                size_hint_y: 0.8
            Button:
                id: path_button
                text: 'Save project path'
                on_release: root.open_filechooser()
                size_hint_y: 0.8
        PrevNextButtons:
            size_hint_y: 0.2
            id: prev_next_buttons
"""

Builder.load_string(kv_string)


class CreateProjectScreen(Screen):
    def __init__(self, **kwargs):
        shared_data = kwargs.pop("shared_data", None)
        super(CreateProjectScreen, self).__init__(**kwargs)

        self.shared_data = shared_data
        self.model_dropdown = self.create_model_dropdown()
        self.ids.prev_next_buttons.on_back = self.go_to_welcome
        self.ids.prev_next_buttons.on_next = self.save_and_go_to_data_set

    def create_model_dropdown(self):
        dropdown = DropDown()
        models = ["BiLSTM", "Model B", "Model C"]  # Example model names
        for model in models:
            btn = Button(text=model, size_hint_y=None, height=44)
            btn.bind(on_release=lambda btn: dropdown.select(btn.text))
            dropdown.add_widget(btn)
        dropdown.bind(
            on_select=lambda instance, x: setattr(
                self.ids.model_button, "text", x
            )
        )
        return dropdown

    def go_to_welcome(self):
        self.shared_data = ProjectData()
        self.ids.name_input.text = ""
        self.ids.model_button.text = "Choose a model"
        self.ids.description_input.text = ""
        self.ids.path_button.text = "Save project path"
        self.manager.current = "welcome"

    def open_filechooser(self):
        filechooser = FileChooserIconView(dirselect=True)
        layout = BoxLayout(orientation="vertical")
        layout.add_widget(filechooser)
        button = Button(text="OK", size_hint=(1, 0.2))
        layout.add_widget(button)
        popup = Popup(
            title="Choose folder", content=layout, size_hint=(0.9, 0.9)
        )
        button.bind(
            on_release=lambda x: self.select_path(
                filechooser, filechooser.selection, popup
            )
        )
        popup.open()

    def select_path(self, instance, selection, popup):
        if selection:
            project_path = create_unique_folder_name(
                selection[0], self.ids.name_input.text
            )
            self.ids.path_button.text = selection[0] + "/" + project_path
            popup.dismiss()

    def save_and_go_to_data_set(self):
        name = self.ids.name_input.text.strip()
        description = self.ids.description_input.text.strip()
        model = self.ids.model_button.text.strip()
        path = self.ids.path_button.text.strip()

        if name and description and path != "Save project path":
            self.shared_data.name = name
            self.shared_data.model = model
            self.shared_data.description = description
            self.shared_data.save_path = path
            self.manager.current = "data_set"
        else:
            popup = Popup(
                title="Error",
                content=Label(text="You have to enter all required fields."),
                size_hint=(None, None),
                size=(300, 200),
            )
            popup.open()
