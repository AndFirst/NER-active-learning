from kivy.uix.screenmanager import Screen
from kivy.lang import Builder
from kivy.uix.label import Label
from app.project import Project
from app.data_types import LabelData

kv_string = """
<SummaryScreen>:
    BoxLayout:
        id: box_layout
        color: (1, 0, 0, 1)  
        orientation: 'vertical'
        Label:
            text: "Summary"
            font_size: 24
            size_hint_y: 0.4
            color: (0, 0, 0, 1)
            halign: 'center'
            valign: 'top'

        GridLayout: 
            id: field_value_grid
            cols: 2
            spacing: [5, 5]  # Add spacing between cells
            row_default_height: 40
            row_force_default: True
            size_hint_y: None
            height: self.minimum_height

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
"""

Builder.load_string(kv_string)


class CustomLabel(Label):
    pass


class SummaryScreen(Screen):
    def __init__(self, **kwargs):
        form_state = kwargs.pop("form_state", None)
        super(SummaryScreen, self).__init__(**kwargs)
        self.form_state = form_state

    def on_enter(self):
        data_labels = self.ids.field_value_grid.children[2:]
        for label in data_labels:
            self.ids.field_value_grid.remove_widget(label)

        self.ids.prev_next_buttons.on_back = self.go_to_add_labels
        self.ids.prev_next_buttons.on_next = self.go_to_main_menu
        self.populate_field_values()

    def populate_field_values(self):
        """Populates the field-value grid layout."""
        grid_layout = self.ids.field_value_grid
        field_labels = ["Name", "Description", "Save Path", "Dataset Path", "Labels"]
        values = [
            self.form_state.name,
            self.form_state.description,
            self.form_state.save_path.split("/")[-3:],
            self.form_state.dataset_path.split("/")[-3:],
            self.form_state.labels
        ]

        for field_label_text, value in zip(field_labels, values):
            field_label = CustomLabel(
                text=field_label_text, halign="left", color=(0, 0, 0, 1),
                size_hint_x=0.25
            )
            if field_label_text in ["Dataset Path", "Save Path"]:
                value_label = CustomLabel(
                    text=".../" + str("/").join(value),
                    halign="right",
                    color=(0, 0, 0, 1),
                    size_hint_x=0.75  
                )
            elif field_label_text == "Description":
                wrapped_text = ""
                for i in range(0, len(value), 50):
                    wrapped_text += value[i: i + 50] + "\n"
                value_label = CustomLabel(
                    text=wrapped_text, halign="right", color=(0, 0, 0, 1),
                    size_hint_x=0.75  
                )
            elif field_label_text == "Labels":
                labels_text = ", ".join(f"{label_data.label} ({label_data.color})" for label_data in value)
                value_label = CustomLabel(
                    text=labels_text, halign="right", color=(0, 0, 0, 1),
                    size_hint_x=0.75  
                )
            # elif field_label_text == "Labels":
            #     for label_data in value:
            #         label_text = f"{label_data.label}"
            #         label_color = label_data.color
            #         label_widget = CustomLabel(
            #             text=label_text, halign="right", color=label_color,
            #             size_hint_x=0.75  
            #         )
            #         grid_layout.add_widget(CustomLabel(text="", size_hint_x=0.25))
            #         grid_layout.add_widget(label_widget)
            #     continue
            else:
                value_label = CustomLabel(
                    text=str(value), halign="right", color=(0, 0, 0, 1),
                    size_hint_x=0.75
                )
            grid_layout.add_widget(field_label)
            grid_layout.add_widget(value_label)

    def go_to_add_labels(self):
        self.manager.current = "add_labels"

    def go_to_main_menu(self):
        state = self.form_state.to_dict()
        Project.create(self.form_state.save_path, state)
        self.manager.get_screen("main_menu").project = Project.load(
            self.form_state.save_path
        )
        self.manager.get_screen("main_menu").save_path = (
            self.form_state.save_path
        )
        self.manager.current = "main_menu"
