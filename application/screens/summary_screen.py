from kivy.uix.screenmanager import Screen
from kivy.lang import Builder

kv_string = """
<SummaryScreen>:
    BoxLayout:
        orientation: 'vertical'
        Label:
            text: 'Podsumowanie projektu:'
        PrevNextButtons:
            id: prev_next_buttons
            back_text: "Wstecz"
            next_text: "Dalej"
"""
Builder.load_string(kv_string)


class SummaryScreen(Screen):
    def __init__(self, **kwargs):
        shared_data = kwargs.pop('shared_data', None)
        super(SummaryScreen, self).__init__(**kwargs)
        self.shared_data = shared_data

        self.ids.prev_next_buttons.on_back = self.go_to_add_labels
        self.ids.prev_next_buttons.on_next = self.go_to_main_menu

    def go_to_add_labels(self):
        self.manager.current = 'add_labels'

    def go_to_main_menu(self):
        self.manager.current = 'main_menu'
