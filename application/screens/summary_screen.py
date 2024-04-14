import copy
import json

from kivy.app import App
from kivy.uix.screenmanager import Screen
from kivy.lang import Builder
import os

from application.file_operations import save_project

kv_string = """
<SummaryScreen>:
    BoxLayout:
        orientation: 'vertical'
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
        shared_data = kwargs.pop('shared_data', None)
        super(SummaryScreen, self).__init__(**kwargs)
        self.shared_data = shared_data

        self.ids.prev_next_buttons.on_back = self.go_to_add_labels
        self.ids.prev_next_buttons.on_next = self.go_to_main_menu

    def go_to_add_labels(self):
        self.manager.current = 'add_labels'

    def go_to_main_menu(self):
        print('L2', self.shared_data.labels)
        data_copy = copy.copy(self.shared_data)
        save_project(data_copy, 'saved_projects/')
        self.manager.current = 'main_menu'
