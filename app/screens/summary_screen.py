from kivy.uix.screenmanager import Screen
from kivy.lang import Builder
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from file_operations import save_project

kv_string = """
<SummaryScreen>:
    BoxLayout:
        id: box_layout
        color: (1, 0, 0, 1)  
        orientation: 'vertical'
        Label:
            text: "Summary"
            font_size: 24  
            halign: 'center'  
            valign: 'top'
        GridLayout:  
            cols: 2
            Label:
                text: "Field"  
                font_size: 20 
                bold: True 
                halign: 'left'
            Label:
                text: "Value"  
                font_size: 20 
                bold: True 
                halign: 'right'  
        PrevNextButtons:
            id: prev_next_buttons
            size_hint_y: 0.2
"""
Builder.load_string(kv_string)


# @TODO Write summarising of input data - ZUZIA
# @TODO on "Dalej" check if data is correct. - IREK
class SummaryScreen(Screen):
    def __init__(self, **kwargs):
        shared_data = kwargs.pop("shared_data", None)
        super(SummaryScreen, self).__init__(**kwargs)
        self.shared_data = shared_data
        self.ids.prev_next_buttons.on_back = self.go_to_add_labels
        self.ids.prev_next_buttons.on_next = self.go_to_main_menu

        data_label = Label(text=str(self.shared_data), size_hint_y=None)
        data_label.bind(width=lambda s, w: s.setter("text_size")(s, (w, None)))
        scroll_view = ScrollView()
        scroll_view.add_widget(data_label)
        self.ids.box_layout.add_widget(scroll_view)

    def go_to_add_labels(self):
        self.manager.current = "add_labels"

    def go_to_main_menu(self):
        save_project(self.shared_data, "saved_projects/")
        self.manager.current = "main_menu"

    def on_enter(self):
        grid_layout = self.ids.box_layout.children[1]

        # Usuwamy wszystkie istniejące widgety z GridLayout
        grid_layout.clear_widgets()

        # Lista etykiet pól
        field_labels = ["Name", "Description", "Save Path", "Dataset Path"]

        # Pobieramy wartości z obiektu ProjectData
        values = [
            self.shared_data.name,
            self.shared_data.description,
            self.shared_data.save_path,
            self.shared_data.dataset_path,
        ]

        # Dodajemy labelki dla pól i wartości
        for field_label_text, value in zip(field_labels, values):
            field_label = Label(text=field_label_text, halign="left")
            value_label = Label(text=str(value), halign="left")
            grid_layout.add_widget(field_label)
            grid_layout.add_widget(value_label)

        # Dodajemy labelki dla etykiet
        label_title = Label(
            text="Labels", font_size=20, bold=True, halign="left"
        )
        grid_layout.add_widget(label_title)
        for label_data in self.shared_data.labels:
            label_name = label_data.label
            label_color = label_data.color
            label_info = f"Name: {label_name}, Color: {label_color}"
            label_info_label = Label(text=label_info, halign="left")
            grid_layout.add_widget(label_info_label)
