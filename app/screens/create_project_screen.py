from kivy.uix.screenmanager import Screen
from kivy.lang import Builder
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.dropdown import DropDown
from kivy.uix.textinput import TextInput

from app.data_types import ProjectFormState
from plyer import filechooser
from app.utils import create_unique_folder_name

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
                on_text_validate: root.update_save_button_text(self.text)
            MaxLengthInput:
                id: description_input
                hint_text: 'Description'
            Button:
                id: model_button
                text: 'Choose a model'
                on_release: root.model_dropdown.open(self)
                size_hint_y: 0.8
            Button:
                id: state_button
                text: 'Load model state'
                on_release: root.open_state_filechooser()
                size_hint_y: 0.8
            Button:
                id: path_button
                text: 'Save project path'
                on_release: root.open_filechooser()
                size_hint_y: 0.8
            Button:
                id: output_type_button
                text: 'Choose output file type'
                on_release: root.output_type_dropdown.open(self)
                size_hint_y: 0.8
        PrevNextButtons:
            size_hint_y: 0.2
            id: prev_next_buttons
"""

Builder.load_string(kv_string)

class MaxLengthInput(TextInput):
    max_length = 200

    def insert_text(self, substring, from_undo=False):
        if len(self.text) + len(substring) <= self.max_length:
            return super().insert_text(substring, from_undo=from_undo)


class CreateProjectScreen(Screen):
    def __init__(self, **kwargs):
        form_state = kwargs.pop("form_state", None)
        super(CreateProjectScreen, self).__init__(**kwargs)

        self.form_state = form_state
        self.model_dropdown = self.create_dropdown(
            ["BiLSTM", "Your model"], self.handle_model_selection
        )
        self.output_type_dropdown = self.create_dropdown(
            [".csv", ".json"], self.handle_output_type_selection
        )
        self.ids.prev_next_buttons.on_back = self.go_to_welcome
        self.ids.prev_next_buttons.on_next = self.save_and_go_to_data_set

    def create_dropdown(self, options, on_select):
        dropdown = DropDown()
        for option in options:
            btn = Button(text=option, size_hint_y=None, height=44)
            btn.bind(on_release=lambda btn: on_select(btn.text, dropdown))
            dropdown.add_widget(btn)
        return dropdown

    def handle_output_type_selection(self, output_type, dropdown):
        dropdown.select(output_type)
        self.ids.output_type_button.text = output_type
        self.form_state.output_extension = output_type

    def handle_model_selection(self, model_name, dropdown):
        dropdown.select(model_name)
        actions = {
            "Your model": lambda _: self.open_model_filechooser(),
            "BiLSTM": lambda _: self.set_model_button_text(model_name),
        }
        actions.get(model_name, self.set_model_button_text)(model_name)

    def set_model_button_text(self, text):
        self.ids.model_button.text = text
        self.form_state.model_type = text

    def open_model_filechooser(self):
        file_path = filechooser.open_file(
            filters=["*.py"],
            title="Select Model Implementation File",
            multiple=False,
        )
        if file_path:
            selected_path = file_path[0]
            self.form_state.model_implementation_path = selected_path
            self.ids.model_button.text = selected_path
            self.form_state.model_type = "custom"
        else:
            self.ids.model_button.text = "Choose a model"

    def open_state_filechooser(self):
        file_path = filechooser.open_file(
            filters=["*.pth", "*.pt"],
            title="Select Model State File",
            multiple=False,
        )
        if file_path:
            selected_path = file_path[0]
            self.form_state.model_state_path = selected_path
            self.ids.state_button.text = selected_path
        else:
            self.ids.state_button.text = "Load model state"

    def go_to_welcome(self):
        self.form_state = ProjectFormState()
        self.ids.name_input.text = ""
        self.ids.model_button.text = "Choose a model"
        self.ids.description_input.text = ""
        self.ids.path_button.text = "Save project path"
        self.manager.current = "welcome"

    def open_filechooser(self):
        if not self.ids.name_input.text.strip():
            popup = Popup(
                title="Error",
                content=Label(text="You have to enter a project name first."),
                size_hint=(None, None),
                size=(300, 200),
            )
            popup.open()
        else:
            selected_path = filechooser.choose_dir(
                title="Select Project Folder"
            )
            if selected_path:
                folder_path = selected_path[0]
                unique_folder_path = create_unique_folder_name(
                    folder_path, self.ids.name_input.text
                )
                self.ids.path_button.text = (
                    folder_path + "/" + unique_folder_path
                )
            else:
                self.ids.path_button.text = "Save project path"

    def save_and_go_to_data_set(self):
        name = self.ids.name_input.text.strip()
        description = self.ids.description_input.text.strip()
        path = self.ids.path_button.text.strip()
        print(self.form_state)

        if all([
            name,
            description,
            path != "Save project path",
            self.form_state.output_extension,
            self.form_state.model_type
        ]):
            self.form_state.name = name
            self.form_state.description = description
            self.form_state.save_path = path
            self.manager.current = "data_set"
        else:
            popup = Popup(
                title="Error",
                content=Label(text="You have to enter all required fields."),
                size_hint=(None, None),
                size=(300, 200),
            )
            popup.open()

    def update_save_button_text(self, text):
        text = text.strip()
        if not text.strip():
            self.ids.path_button.text = "Save project path"
            return
        if self.ids.path_button.text.strip() != "Save project path":
            folder_path = "/".join(self.ids.path_button.text.split("/")[:-1])
            unique_folder_path = create_unique_folder_name(folder_path, text)
            self.ids.path_button.text = folder_path + "/" + unique_folder_path
