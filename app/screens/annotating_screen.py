from kivy.uix.screenmanager import Screen
from kivy.lang import Builder

from app.components.popups.popups import (
    SaveConfirmationPopup,
)

kv_string = """
<MainMenuScreen>:
    AnnotationForm:
        id: annotation_form        
"""

Builder.load_string(kv_string)


class MainMenuScreen(Screen):
    def __init__(self, **kwargs):
        super(MainMenuScreen, self).__init__(**kwargs)
        self.model = None
        self.project = None
        self.assistant = None
        self.save_path = None

    def on_enter(self):
        self.assistant = self.project.get_assistant()
        self.model = self.project.get_model()
        self.ids.annotation_form.labels = self.project.get_labels()

        self.ids.annotation_form.sentence = self.assistant.get_sentence(
            annotated=True
        )
        self.manager.get_screen("stats").stats = self.assistant.stats

    def confirm_exit(self):
        exit_confirmation_popup = SaveConfirmationPopup(
            save_function=self.save
        )
        exit_confirmation_popup.open()
        return True

    def save(self):
        self.project.save()
