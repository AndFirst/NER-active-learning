from kivy.uix.screenmanager import Screen
from kivy.lang import Builder

from app.components.popups.popups import (
    SaveConfirmationPopup,
)
from app.data_types import LabelData

kv_string = """
<MainMenuScreen>:
    BoxLayout:
        orientation: 'horizontal'
        GridLayout:
            cols: 1
            size_hint_x: 0.2
            pos_hint: {'top':1}
            color: 0, 0, 0, 1
            Button:
                text: 'Annotate'
            Button:
                text: 'Dataset'
            Button:
                text: 'Labels'
            Button:
                text: 'Stats' 
                on_release: app.root.current = 'stats'
        AnnotationForm:
            id: annotation_form        
            size_hint_x: 0.8
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
        labels = self.project.get_labels()
        self.ids.annotation_form.labels = self._init_ui_labels(labels)

        self.ids.annotation_form.sentence = self.assistant.get_sentence(
            annotated=True
        )
        self.manager.get_screen("stats").stats = self.assistant.stats

    def _init_ui_labels(self, label_data: dict) -> list:
        return [LabelData(label, color) for label, color in label_data.items()]

    def confirm_exit(self):
        exit_confirmation_popup = SaveConfirmationPopup(
            save_function=self.save
        )
        exit_confirmation_popup.open()
        return True

    def save(self):
        self.project.save(self.save_path)
