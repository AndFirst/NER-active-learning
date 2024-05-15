from kivy.app import App
from kivy.uix.screenmanager import ScreenManager
from app.data_types import ProjectFormState
from app.components.add_labels.add_label_form import AddLabelForm
from app.components.annotating.annotation_container import AnnotationContainer
from app.components.annotating.annotation_form import AnnotationForm
from app.components.annotating.label_choose_container import (
    LabelChooseContainer,
)
from app.components.add_labels.label_row import LabelRow
from app.components.annotating.label import ColorLabel
from app.components.prev_next_buttons import PrevNextButtons
from app.components.annotating.token import Token
from app.screens import (
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


class Application(App):
    def build(self):
        Window.clearcolor = BACKGROUND_COLOR

        project_form_state = ProjectFormState()
        sm = ScreenManager()
        sm.add_widget(WelcomeScreen(name="welcome"))
        sm.add_widget(ExistingProjectScreen(name="existing_project"))
        sm.add_widget(
            CreateProjectScreen(
                name="create_project", form_state=project_form_state
            )
        )
        sm.add_widget(
            DatasetScreen(name="data_set", form_state=project_form_state)
        )
        sm.add_widget(
            AddLabelsScreen(name="add_labels", form_state=project_form_state)
        )
        sm.add_widget(
            SummaryScreen(name="summary", form_state=project_form_state)
        )
        sm.add_widget(MainMenuScreen(name="main_menu"))
        inspector.create_inspector(Window, sm)
        return sm

    def on_start(self):
        self.title = "Text Annotating App"

    @property
    def home_dir(self) -> str:
        return str(Path(__file__).resolve())


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
