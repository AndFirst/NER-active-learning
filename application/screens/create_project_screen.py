from kivy.uix.screenmanager import Screen
from kivy.lang import Builder
from kivy.uix.popup import Popup
from kivy.uix.label import Label

from application.data_types import ProjectData

kv_string = """
<CreateProjectScreen>:
    BoxLayout:
        orientation: 'vertical'
        size_hint: (1, 1)
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
            size_hint_y: 0.8
            Label:
                color: 0, 0, 0, 1
                text: 'Insert project data:'
            BoxLayout:
                orientation: 'vertical'
                padding: (100, 10)
                spacing: 20
                TextInput:
                    id: name_input
                    hint_text: 'Name'
                    multiline: False
                TextInput:
                    id: description_input
                    hint_text: 'Description'
        PrevNextButtons:
            size_hint_y: 0.2
            id: prev_next_buttons
"""

Builder.load_string(kv_string)


# @TODO Add more project data inputs,  - ZUZIA
#   1. Add save project path
#   2. Clean up interface
class CreateProjectScreen(Screen):
    def __init__(self, **kwargs):
        shared_data = kwargs.pop('shared_data', None)
        super(CreateProjectScreen, self).__init__(**kwargs)

        self.shared_data = shared_data
        self.ids.prev_next_buttons.on_back = self.go_to_welcome
        self.ids.prev_next_buttons.on_next = self.save_and_go_to_data_set

    def go_to_welcome(self):
        self.shared_data = ProjectData()
        self.ids.name_input.text = ''
        self.ids.description_input.text = ''
        self.manager.current = 'welcome'

    def save_and_go_to_data_set(self):
        name = self.ids.name_input.text.strip()
        description = self.ids.description_input.text.strip()

        if name and description:
            self.shared_data.name = name
            self.shared_data.description = description
            self.manager.current = 'data_set'
        else:
            popup = Popup(title='Error',
                          content=Label(
                              text='You have to enter all required fields.'),
                          size_hint=(None, None), size=(300, 200))
            popup.open()
