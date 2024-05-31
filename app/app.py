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
    StatsScreen,
)
from app.ui_colors import BACKGROUND_COLOR
from kivy.core.window import Window
from pathlib import Path


class Application(App):
    """
    The main application class that initializes the screen manager and the screens.

    :param kwargs: Keyword arguments passed to the Kivy App initializer.
    :type kwargs: dict
    """

    def __init__(self, **kwargs):
        super(Application, self).__init__(**kwargs)
        self._sm = ScreenManager()

    def build(self):
        """
        Build the application by setting the window color and binding the close event.

        :return: The screen manager.
        :rtype: kivy.uix.screenmanager.ScreenManager
        """
        Config.set("input", "mouse", "mouse,multitouch_on_demand")
        Window.clearcolor = BACKGROUND_COLOR
        Window.bind(on_request_close=self.on_request_close)

        project_form_state = ProjectFormState()
        self._sm.add_widget(WelcomeScreen(name="welcome"))
        self._sm.add_widget(ExistingProjectScreen(name="existing_project"))
        self._sm.add_widget(CreateProjectScreen(name="create_project", form_state=project_form_state))
        self._sm.add_widget(DatasetScreen(name="data_set", form_state=project_form_state))
        self._sm.add_widget(AddLabelsScreen(name="add_labels", form_state=project_form_state))
        self._sm.add_widget(SummaryScreen(name="summary", form_state=project_form_state))
        self._sm.add_widget(MainMenuScreen(name="main_menu"))
        self._sm.add_widget(StatsScreen(name="stats"))
        return self._sm

    def on_start(self):
        """
        Set the title of the application when it starts.
        """
        self.title = "Text Annotating App"

    @property
    def home_dir(self) -> str:
        """
        Get the path of the current file.

        :return: The path of the current file.
        :rtype: str
        """
        return str(Path(__file__).resolve())

    def on_request_close(self, *args, **kwargs) -> bool:
        """
        Handle the close event by calling the confirm_exit method.

        :param args: Positional arguments passed to the on_request_close method.
        :type args: tuple

        :param kwargs: Keyword arguments passed to the on_request_close method.
        :type kwargs: dict
        :return: True if the application should close, False otherwise.
        :rtype: bool
        """
        current_screen = self._sm.current_screen
        if hasattr(current_screen, "confirm_exit"):
            return current_screen.confirm_exit()
        else:
            return self._default_confirm_exit() or True

    def _default_confirm_exit(self):
        exit_confirmation_popup = ExitConfirmationPopup()
        exit_confirmation_popup.open()

    @property
    def manager(self):
        return self._sm


def init_imports():
    """
    Don't remove this function. It is used to initialize the imports for the application.
    Without this function, kivy components will not be loaded properly.
    """
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
