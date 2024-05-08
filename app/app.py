from kivy.app import App
from kivy.uix.screenmanager import ScreenManager
from app.data_types import ProjectData
from app.components.add_label_form import AddLabelForm
from app.components.annotation_container import AnnotationContainer
from app.components.annotation_form import AnnotationForm
from app.components.label_choose_container import LabelChooseContainer
from app.components.label_row import LabelRow
from app.components.label import ColorLabel
from app.components.prev_next_buttons import PrevNextButtons
from app.components.token import Token
from screens import (
    WelcomeScreen,
    ExistingProjectScreen,
    CreateProjectScreen,
    MainMenuScreen,
    DatasetScreen,
    AddLabelsScreen,
    SummaryScreen,
)
from app.ui_colors import BACKGROUND_COLOR
from kivy.core.window import Window
from pathlib import Path
from kivy.modules import inspector


# @TODO Integrate both apps: - IREK
#   1. Chosen labels and their colors should be visible in annotation screen.
# @TODO Justfile/Makefile - RAFAŁ
# @TODO setup unit tests - RAFAŁ
# @TODO Placeholder for stats - ZUZIA
# @TODO Labels preview on main screen - ZUZIA
# @TODO Dataset preview on main screen - ZUZIA


class Application(App):
    def build(self):
        Window.clearcolor = BACKGROUND_COLOR

        self.home_dir = str(Path(__file__).resolve())
        shared_data = ProjectData()
        sm = ScreenManager()
        sm.add_widget(WelcomeScreen(name="welcome"))
        sm.add_widget(
            ExistingProjectScreen(
                name="existing_project", shared_data=shared_data
            )
        )
        sm.add_widget(
            CreateProjectScreen(name="create_project", shared_data=shared_data)
        )
        sm.add_widget(DatasetScreen(name="data_set", shared_data=shared_data))
        sm.add_widget(
            AddLabelsScreen(name="add_labels", shared_data=shared_data)
        )
        sm.add_widget(SummaryScreen(name="summary", shared_data=shared_data))
        sm.add_widget(
            MainMenuScreen(name="main_menu", shared_data=shared_data)
        )
        inspector.create_inspector(Window, sm)
        return sm

    def on_start(self):
        self.title = "Give me a name"


def init_imports():
    AddLabelForm
    AnnotationContainer
    AnnotationForm
    LabelChooseContainer
    LabelRow
    ColorLabel
    PrevNextButtons
    Token


if __name__ == "__main__":
    Application().run()
