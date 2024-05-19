from kivy.uix.screenmanager import Screen
from kivy.lang import Builder
from kivy.uix.label import Label

from app.project import Project

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
            Label:
                text: "Field"
                font_size: 20
                bold: True
                halign: 'left'
                color: (0, 0, 0, 1)
            Label:
                text: "Value"
                font_size: 20
                bold: True
                halign: 'right'
                color: (0, 0, 0, 1)

        PrevNextButtons:
            id: prev_next_buttons
            size_hint_y: None
            valign: 'bottom'

"""
Builder.load_string(kv_string)


class SummaryScreen(Screen):
    def __init__(self, **kwargs):
        form_state = kwargs.pop("form_state", None)
        super(SummaryScreen, self).__init__(**kwargs)
        self.form_state = form_state

    def on_enter(self):
        field_label = Label(
            text="Field",
            font_size=20,
            bold=True,
            halign="left",
            color=(0, 0, 0, 1),
        )
        value_label = Label(
            text="Value",
            font_size=20,
            bold=True,
            halign="right",
            color=(0, 0, 0, 1),
        )

        self.ids.field_value_grid.add_widget(field_label)
        self.ids.field_value_grid.add_widget(value_label)
        data_labels = self.ids.field_value_grid.children[2:]
        for label in data_labels:
            self.ids.field_value_grid.remove_widget(label)

        self.ids.prev_next_buttons.on_back = self.go_to_add_labels
        self.ids.prev_next_buttons.on_next = self.go_to_main_menu
        self.populate_field_values()

    def populate_field_values(self):
        """Populates the field-value grid layout."""
        grid_layout = self.ids.field_value_grid
        field_labels = ["Name", "Description", "Save Path", "Dataset Path"]
        values = [
            self.form_state.name,
            self.form_state.description,
            self.form_state.save_path.split("/")[-3:],
            self.form_state.dataset_path.split("/")[-3:],
        ]

        for field_label_text, value in zip(field_labels, values):
            field_label = Label(
                text=field_label_text, halign="left", color=(0, 0, 0, 1)
            )
            if (
                field_label_text == "Dataset Path"
                or field_label_text == "Save Path"
            ):
                value_label = Label(
                    text=".../" + str("/".join(value)),
                    halign="right",
                    color=(0, 0, 0, 1),
                )
            elif field_label_text == "Description":
                wrapped_text = ""
                for i in range(0, len(value), 50):
                    wrapped_text += value[i : i + 50] + "\n"
                value_label = Label(
                    text=wrapped_text, halign="right", color=(0, 0, 0, 1)
                )
            else:
                value_label = Label(
                    text=str(value), halign="right", color=(0, 0, 0, 1)
                )
            grid_layout.add_widget(field_label)
            grid_layout.add_widget(value_label)

    def go_to_add_labels(self):
        self.manager.current = "add_labels"

    def go_to_main_menu(self):
        state = {
            "name": self.form_state.name,
            "description": self.form_state.description,
            "labels": [label.to_dict() for label in self.form_state.labels],
            "dataset_path": self.form_state.dataset_path,
            "input_extension": ".csv",
            "output_extension": ".csv",
        }
        Project.create(self.form_state.save_path, state)
        self.manager.get_screen("main_menu").project = Project.load(
            self.form_state.save_path
        )
        self.manager.get_screen("main_menu").save_path = (
            self.form_state.save_path
        )
        self.manager.current = "main_menu"
