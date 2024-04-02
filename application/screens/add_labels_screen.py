# add_labels_screen.py
from kivy.uix.screenmanager import Screen
from kivy.lang import Builder
from components.prev_next_buttons import PrevNextButtons
from components.add_label_form import AddLabelForm
kv_string = """
<AddLabelsScreen>:
    orientation: 'vertical'
    size_hint: (1, 1)
    AddLabelForm:
        id: add_label_form
    PrevNextButtons:
        id: prev_next_buttons
        back_text: "Wstecz"
        next_text: "Dalej"

"""

Builder.load_string(kv_string)


class AddLabelsScreen(Screen):

    def __init__(self, **kwargs):
        shared_data = kwargs.pop('shared_data', None)
        super(AddLabelsScreen, self).__init__(**kwargs)
        self.shared_data = shared_data
        self.ids.prev_next_buttons.on_back = self.go_to_data_set
        self.ids.prev_next_buttons.on_next = self.go_to_summary

    def go_to_data_set(self):
        self.manager.current = 'data_set'

    def go_to_summary(self):
        self.manager.current = 'summary'
