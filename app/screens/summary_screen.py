from kivy.uix.popup import Popup
from kivy.uix.screenmanager import Screen
from kivy.lang import Builder
from kivy.uix.label import Label
from app.data_types import ProjectFormState
from app.project import Project

kv_string = """
<SummaryScreen>:
    BoxLayout:
        id: box_layout
        orientation: 'vertical'
        BoxLayout:
            size_hint_y: 0.1
            Label:
                text: "Summary"
                font_size: 24
                color: (0, 0, 0, 1)
                halign: 'center'
                valign: 'middle'
        GridLayout:
            id: field_value_grid
            cols: 2
            spacing: [0, 0]
            row_default_height: 60
            row_force_default: True
            size_hint_y: None

        PrevNextButtons:
            id: prev_next_buttons
            size_hint_y: None
            valign: 'bottom'

<CustomLabel>:
    canvas.before:
        Color:
            rgba: 0, 0, 0, 1
        Line:
            width: 1
            rectangle: self.x, self.y, self.width, self.height
    size_hint_y: None
    height: 60
    text_size: self.width, None
    halign: 'left'
    valign: 'middle'
"""

Builder.load_string(kv_string)


class CustomLabel(Label):
    pass


class SummaryScreen(Screen):
    def __init__(self, **kwargs):
        form_state = kwargs.pop("form_state", None)
        super(SummaryScreen, self).__init__(**kwargs)
        self.form_state: ProjectFormState = form_state

    def on_enter(self):
        self.ids.field_value_grid.clear_widgets()
        self.ids.prev_next_buttons.on_back = self.go_to_add_labels
        self.ids.prev_next_buttons.on_next = self.go_to_main_menu
        self.populate_field_values()

    def populate_field_values(self):
        """Populates the field-value grid layout."""
        grid_layout = self.ids.field_value_grid
        field_labels = [
            "Name",
            "Description",
            "Save Path",
            "Dataset Path",
            "Labels",
        ]
        values = [
            self.form_state.name,
            self.form_state.description,
            self.form_state.save_path.split("/")[-3:],
            self.form_state.dataset_path.split("/")[-3:],
            self.form_state.labels,
        ]

        for field_label_text, value in zip(field_labels, values):
            field_label = CustomLabel(
                text=field_label_text,
                halign="center",
                color=(0, 0, 0, 1),
                text_size=(20, None),
                size_hint_x=0.25,
            )
            if field_label_text in ["Dataset Path", "Save Path"]:
                value_label = CustomLabel(
                    text=".../" + str("/").join(value),
                    halign="center",
                    color=(0, 0, 0, 1),
                    size_hint_x=0.75,
                )
            elif field_label_text == "Description":
                value_label = CustomLabel(
                    text=value,
                    halign="center",
                    color=(0, 0, 0, 1),
                    size_hint_x=0.75,
                )
            elif field_label_text == "Labels":
                labels_text = ", ".join(
                    f"{label_data.label}" for label_data in value
                )
                value_label = CustomLabel(
                    text=labels_text,
                    halign="center",
                    color=(0, 0, 0, 1),
                    size_hint_x=0.75,
                )
            else:
                value_label = CustomLabel(
                    text=str(value),
                    halign="center",
                    color=(0, 0, 0, 1),
                    size_hint_x=0.75,
                )
            grid_layout.add_widget(field_label)
            grid_layout.add_widget(value_label)

        grid_layout.bind(minimum_height=grid_layout.setter("height"))

    def go_to_add_labels(self):
        self.manager.current = "add_labels"

    def go_to_main_menu(self):
        try:
            project = Project.create(self.form_state)
            project.save()
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
        self.manager.get_screen("main_menu").project = Project.load(
            self.form_state.save_path
        )
        self.manager.get_screen("main_menu").save_path = (
            self.form_state.save_path
        )
        self.manager.current = "main_menu"
