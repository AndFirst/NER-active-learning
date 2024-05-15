from kivy.uix.screenmanager import Screen
from kivy.lang import Builder
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.dropdown import DropDown

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
        form_state = kwargs.pop("form_state", None)
        super(CreateProjectScreen, self).__init__(**kwargs)

        self.form_state = form_state
        self.model_dropdown = self.create_model_dropdown()
        self.ids.prev_next_buttons.on_back = self.go_to_welcome
        self.ids.prev_next_buttons.on_next = self.save_and_go_to_data_set

    def create_model_dropdown(self):
        dropdown = DropDown()
        models = ["BiLSTM", "Your model"]  # Example model names
        for model in models:
            btn = Button(text=model, size_hint_y=None, height=44)
            btn.bind(
                on_release=lambda btn: self.handle_model_selection(
                    btn.text, dropdown
                )
            )
            dropdown.add_widget(btn)
        return dropdown

    def handle_model_selection(self, model_name, dropdown):
        dropdown.select(model_name)
        if model_name == "Your model":
            self.open_model_filechooser()
        else:
            self.ids.model_button.text = model_name

    def open_model_filechooser(self):
        # Use plyer's filechooser to open the native file dialog
        file_path = filechooser.open_file(
            filters=["*.pth"], title="Select Model File", multiple=False
        )
        if file_path:
            selected_path = file_path[
                0
            ]  # Since 'multiple=False', it returns a list with one item
            self.ids.model_button.text = selected_path
            self.form_state["user_model_path"] = selected_path
        else:
            self.ids.model_button.text = (
                "Choose a model"  # Reset if no file is selected
            )

    def select_model_path(self, filechooser, popup):
        selection = filechooser.selection
        if selection:
            self.ids.model_button.text = selection[
                0
            ]  # Update the button text to show the selected path
            popup.dismiss()
        else:
            self.ids.model_button.text = (
                "Choose a model"  # Reset if no file is selected
            )
            popup.dismiss()

    def go_to_welcome(self):
        self.form_state = ProjectFormState()
        self.ids.name_input.text = ""
        self.ids.model_button.text = "Choose a model"
        self.ids.description_input.text = ""
        self.ids.path_button.text = "Save  project path"
        self.manager.current = "welcome"

    def open_filechooser(self):
        # Use plyer's filechooser to open the native directory chooser
        selected_path = filechooser.choose_dir(title="Select Project Folder")
        if selected_path:
            folder_path = selected_path[
                0
            ]  # Since 'choose_dir' returns a list with one item
            unique_folder_path = create_unique_folder_name(
                folder_path, self.ids.name_input.text
            )
            self.ids.path_button.text = folder_path + "/" + unique_folder_path
        else:
            self.ids.path_button.text = (
                "Save project path"  # Reset if no folder is selected
            )

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
            self.form_state.name = name
            self.form_state.model = model
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
