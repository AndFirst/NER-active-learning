from kivy.config import Config
from kivy.app import App
from kivy.uix.screenmanager import ScreenManager
from app.components.popups.popups import ExitConfirmationPopup
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


class Application(App):
    def __init__(self, **kwargs):
        super(Application, self).__init__(**kwargs)
        self._sm = ScreenManager()

    def build(self):
        Config.set("input", "mouse", "mouse,multitouch_on_demand")
        Window.clearcolor = BACKGROUND_COLOR
        Window.bind(on_request_close=self.on_request_close)

        project_form_state = ProjectFormState()
        self._sm.add_widget(WelcomeScreen(name="welcome"))
        self._sm.add_widget(ExistingProjectScreen(name="existing_project"))
        self._sm.add_widget(
            CreateProjectScreen(
                name="create_project", form_state=project_form_state
            )
        )
        self._sm.add_widget(
            DatasetScreen(name="data_set", form_state=project_form_state)
        )
        self._sm.add_widget(
            AddLabelsScreen(name="add_labels", form_state=project_form_state)
        )
        self._sm.add_widget(
            SummaryScreen(name="summary", form_state=project_form_state)
        )
        self._sm.add_widget(MainMenuScreen(name="main_menu"))
        return self._sm

    def on_start(self):
        self.title = "Text Annotating App"

    @property
    def home_dir(self) -> str:
        return str(Path(__file__).resolve())

    def on_request_close(self, *args, **kwargs):
        current_screen = self._sm.current_screen
        if hasattr(current_screen, "confirm_exit"):
            return current_screen.confirm_exit()
        else:
            return self._default_confirm_exit() or True

    def _default_confirm_exit(self):
        exit_confirmation_popup = ExitConfirmationPopup()
        exit_confirmation_popup.open()


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
