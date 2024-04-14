from kivy.uix.screenmanager import Screen
from kivy.lang import Builder
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView

from kivy.uix.gridlayout import GridLayout  # Import GridLayout

from file_operations import save_project

kv_string = """
<SummaryScreen>:
    BoxLayout:
        id: box_layout
        orientation: 'vertical'
        Label:
            text: "Podsumowanie projektu:"
            font_size: 24  
            color: (0, 0, 0, 1)  
            halign: 'center'  
            valign: 'top'
        GridLayout:  
            cols: 2
            Label:
                text: "Pole"  
                font_size: 20 
                bold: True 
                halign: 'left'
            Label:
                text: "Wartość"  
                font_size: 20 
                bold: True 
                halign: 'right'  
        BoxLayout:
            size_hint_y: 0.8
            orientation: 'vertical'
            Label:
                text: 'Summary:'
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

        # Tworzenie etykiety do wyświetlania shared_data
        data_label = Label(text=str(self.shared_data), size_hint_y=None)
        data_label.bind(width=lambda s, w: s.setter("text_size")(s, (w, None)))

        # Dodanie etykiety do ScrollView, aby umożliwić przewijanie, jeśli dane są zbyt duże
        scroll_view = ScrollView()
        scroll_view.add_widget(data_label)

        # Dodanie ScrollView do BoxLayout
        self.ids.box_layout.add_widget(scroll_view)

    def go_to_add_labels(self):
        self.manager.current = "add_labels"

    def go_to_main_menu(self):
        save_project(self.shared_data, "saved_projects/")
        self.manager.current = "main_menu"

    def on_enter(self):
        grid_layout = self.ids.box_layout.children[1]

        for field, value in self.shared_data.get_dict().items():
            field_label = Label(text=field, halign="left")
            value_label = Label(text=value, halign="right")
            grid_layout.add_widget(field_label)
            grid_layout.add_widget(value_label)
