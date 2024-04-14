from kivy.app import App
from kivy.uix.screenmanager import ScreenManager
from screens import (WelcomeScreen, ExistingProjectScreen, CreateProjectScreen,
                     MainMenuScreen, DatasetScreen, AddLabelsScreen, SummaryScreen)
from ui_colors import BACKGROUND_COLOR
from kivy.core.window import Window
from pathlib import Path
from kivy.modules import inspector


# @TODO Move SharedData to another file <CLEANUP> - RAFAŁ
# @TODO Translate all into English - ALL
# @TODO Integrate both apps: - IREK
#   1. Chosen labels and their colors should be visible in annotation screen.
# @TODO Justfile/Makefile - RAFAŁ
# @TODO setup unit tests - RAFAŁ
# @TODO Placeholder for stats - ZUZIA
# @TODO Labels preview on main screen - ZUZIA
# @TODO Dataset preview on main screen - ZUZIA
# @TODO SharedData to defaultdict - RAFAŁ
# @TODO Empty shared data when exit from "create project path" - IREK
class SharedData:
    def __init__(self):
        self.data = {}

    def set_data(self, key, value):
        self.data[key] = value

    def get_data(self, key):
        return self.data.get(key)
    
    def get_dict(self):
        return self.data


class Application(App):
    def build(self):
        Window.clearcolor = BACKGROUND_COLOR

        self.home_dir = str(Path(__file__).resolve())
        shared_data = SharedData()
        sm = ScreenManager()
        sm.add_widget(WelcomeScreen(name='welcome'))
        sm.add_widget(ExistingProjectScreen(name='existing_project'))
        sm.add_widget(CreateProjectScreen(
            name='create_project', shared_data=shared_data))
        sm.add_widget(DatasetScreen(name='data_set', shared_data=shared_data))
        sm.add_widget(AddLabelsScreen(
            name='add_labels', shared_data=shared_data))
        sm.add_widget(SummaryScreen(name='summary', shared_data=shared_data))
        sm.add_widget(MainMenuScreen(name='main_menu'))
        inspector.create_inspector(Window, sm)
        return sm

    def on_start(self):
        self.title = 'Give me a name'


if __name__ == '__main__':
    Application().run()
