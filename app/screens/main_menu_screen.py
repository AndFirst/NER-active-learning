from kivy.core.window import Window
from kivy.uix.screenmanager import Screen
from kivy.lang import Builder
from ui_colors import BACKGROUND_COLOR

from active_learning import ActiveLearningManager

from learning.models.lstm import BiLSTMClassifier

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
        AnnotationForm:
            id: annotation_form        
            size_hint_x: 0.8
"""

Builder.load_string(kv_string)


class MainMenuScreen(Screen):
    def __init__(self, **kwargs):
        Window.clearcolor = BACKGROUND_COLOR
        shared_data = kwargs.pop("shared_data", None)
        super(MainMenuScreen, self).__init__(**kwargs)
        self.shared_data = shared_data
        self.model = None
        self.assistant = None

    def on_enter(self):
        # self.model = self.shared_data.model
        self.model = BiLSTMClassifier(num_words=35178, num_classes=7)
        labeled_path = self.shared_data.save_path + "/labeled.csv"
        unlabeled_path = self.shared_data.save_path + "/unlabeled.csv"
        word_to_idx_path = self.shared_data.save_path + "/word_to_vec.json"
        label_to_idx_path = self.shared_data.save_path + "/label_to_vec.json"

        self.assistant = ActiveLearningManager(
            labeled_path=labeled_path,
            unlabeled_path=unlabeled_path,
            word_to_idx_path=word_to_idx_path,
            label_to_idx_path=label_to_idx_path,
            label_mapping={
                label_data.label: label_data.color
                for label_data in self.shared_data.labels
            },
            model=self.model,
        )
        self.ids.annotation_form.labels = self.shared_data.labels
        self.ids.annotation_form.save_annotation_path = (
            self.shared_data.save_path + "/labeled.csv"
        )

        self.ids.annotation_form.sentence = self.assistant.get_sentence()
