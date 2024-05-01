import csv
from collections import deque

from kivy.core.window import Window
from kivy.uix.screenmanager import Screen
from kivy.lang import Builder
from data_types import (
    Annotation,
    Sentence,
    Word,
)
from ui_colors import BACKGROUND_COLOR

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
        self.sentences = deque()

    def gen_sentence(self):
        if len(self.sentences) > 0:
            yield self.sentences.popleft()

    def on_enter(self):
        self.ids.annotation_form.labels = self.shared_data.labels
        self.ids.annotation_form.save_annotation_path = (
            self.shared_data.save_path + "/labeled.csv"
        )
        sentences = self.read_sentences_from_csv(
            self.shared_data.save_path + "/unlabeled.csv"
        )
        for sentence in sentences:
            annotations = [
                Annotation(words=[Word(word)], label=None) for word in sentence
            ]
            self.sentences.append(Sentence(tokens=annotations))
        self.ids.annotation_form.sentence = self.sentences.popleft()

    def read_sentences_from_csv(self, csv_file_path):
        sentences = []
        with open(csv_file_path, "r") as file:
            reader = csv.reader(file, delimiter="\t")
            for row in reader:
                sentences.append(row)
        return sentences
