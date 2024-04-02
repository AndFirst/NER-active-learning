from kivy.app import App
from kivy.uix.screenmanager import ScreenManager
from screens import (WelcomeScreen, ExistingProjectScreen, CreateProjectScreen,
                     MainMenuScreen,  DatasetScreen, AddLabelsScreen, SummaryScreen)
from ui_colors import BACKGROUND_COLOR
from kivy.core.window import Window
from pathlib import Path
from kivy.modules import inspector


class SharedData:
    def __init__(self):
        self.data = {}

    def set_data(self, key, value):
        self.data[key] = value

    def get_data(self, key):
        return self.data.get(key)


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
