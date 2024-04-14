from collections import deque

from kivy.lang import Builder
from kivy.properties import ObjectProperty, ListProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button

from application.components.label import ColorLabel
from application.components.label_choose_container import LabelChooseContainer
from application.components.annotation_container import AnnotationContainer
from kivy.config import Config

from application.components.token import Token
from application.data_types import Annotation, Sentence, Word
from application.utils import is_key_pressed

Config.set("input", "mouse", "mouse,multitouch_on_demand")

kv_string = """
<AnnotationForm>:
    spacing: 10
    padding: 20
    orientation: 'vertical'
    LabelChooseContainer:
        size_hint_y: 0.3
        id: choose_container
        label_callback: root.update_selected_label
    BoxLayout:
        id: current_annotation
        orientation: 'horizontal'
        size_hint_y: 0.1
        canvas.before:
            Color:
                rgba: 0, 0, 0, 1 
            Line:
                rectangle: (self.x, self.y, self.width, self.height)
                width: 1 
    AnnotationContainer:
        id: annotation_container
        size_hint_y: 0.6
        annotation_callback: root.update_annotation
    BoxLayout:
        id: buttons 
        orientation: 'horizontal'
        size_hint_y: 0.1
        canvas.before:
            Color:
                rgba: 0, 0, 0, 1 
            Line:
                rectangle: (self.x, self.y, self.width, self.height)
                width: 1 
        Button:
            text: 'Accept'
            font_size: '25sp'
            background_color: 0, 1, 0, 1  
        Button:
            text: 'Reset'
            font_size: '25sp'
            background_color: 1, 0, 0, 1 
            on_release: root.reset()
"""

Builder.load_string(kv_string)


# @TODO One label - multiple word - IREK
# @TODO Save on 'Akceptuj' click - IREK
# @TODO Save on press exit button when there is unsaved progress. - IREK
class AnnotationForm(BoxLayout):
    selected_label = ObjectProperty(None, allownone=True)
    labels = ListProperty([])
    sentence = ObjectProperty(None)
    labels_to_merge = ObjectProperty(deque(), allownone=True)
    words = ListProperty([], allownone=True)

    def on_labels(self, instance, value):
        self.ids.choose_container.labels = value

    def on_sentence(self, instance, value):
        self.ids.annotation_container.sentence = value

    def on_selected_label(self, instance, value):
        self.ids.current_annotation.clear_widgets()
        if value is not None:
            color_label = ColorLabel(label_data=value.label_data)
            color_label.on_touch_down = lambda touch: None
            color_label.border_color = color_label.border_color[:3] + [1]
            color_label.selected = 1
            color_label.size_hint = 1, 1
            self.ids.current_annotation.add_widget(color_label)
            self.labels_to_merge = deque()
            for child in self.ids.buttons.children[:]:
                if isinstance(child, Button) and child.text == "Merge":
                    self.ids.buttons.remove_widget(child)

    def update_selected_label(self, new_label: ColorLabel):
        for label_widget in self.ids.choose_container.children:
            if isinstance(label_widget, ColorLabel) and label_widget != new_label:
                label_widget.selected = 0

        new_label.selected = 1 - new_label.selected
        if new_label.selected:
            self.selected_label = new_label
        else:
            self.selected_label = None

    def update_annotation(self, token: Token, touch):
        if self.selected_label and not is_key_pressed('ctrl'):
            if touch.button == 'left':
                token.annotation.label = self.selected_label.label_data
            elif touch.button == 'right':
                token.annotation.label = None
            token.update_label()
            token.canvas.ask_update()

    def reset(self):
        new_tokens = []
        for token in self.sentence.tokens:
            for word in token.words:
                new_token = Annotation(words=[word], label=None)
                new_tokens.append(new_token)
        self.sentence = Sentence(tokens=new_tokens)
        self.selected_label = None
        self.ids.current_annotation.clear_widgets()
        for label in self.ids.choose_container.children:
            label.selected = 0
        self.labels_to_merge = deque()

    def on_labels_to_merge(self, instance, value):
        self.update_labels_to_merge()

    def update_labels_to_merge(self):
        print(len(self.labels_to_merge))
        if len(self.labels_to_merge) == 1 and len(self.ids.buttons.children) == 2:
            # Tworzenie przycisku 'Merge'
            merge_button = Button(text="Merge", size_hint=(None, 1), width=100)
            merge_button.bind(on_release=self.merge_labels)  # Bindowanie przycisku do metody merge_labels
            self.ids.buttons.add_widget(merge_button)  # Dodanie przycisku do kontenera
        elif len(self.labels_to_merge) == 0:
            for child in self.ids.buttons.children[:]:
                if isinstance(child, Button) and child.text == "Merge":
                    self.ids.buttons.remove_widget(child)

    def merge_labels(self, instance):
        if not self.selected_label:
            print("No label selected for merging.")
            return

        merged_words = [Word(label.text) for label in self.labels_to_merge]
        merged_annotation = Annotation(words=merged_words, label=self.selected_label.label_data)

        first_removed_word = self.labels_to_merge[0].word
        index_to_insert = None

        # Znajdź indeks tokenu zawierającego pierwsze słowo do usunięcia
        for index, token in enumerate(self.sentence.tokens):
            if first_removed_word in token.words:
                index_to_insert = index
                if len(token.words) > 1 and not set(token.words).issubset(set(merged_words)):
                    index_to_insert += 1
                break

        if index_to_insert is not None:
            # Usuń etykiety do połączenia
            for label in self.labels_to_merge:
                word_to_remove = label.word
                for token in self.sentence.tokens:
                    if word_to_remove in token.words:
                        token.words.remove(word_to_remove)
                    if not token.words:
                        self.sentence.tokens.remove(token)

            # Wstaw scaloną etykietę
            self.sentence.tokens.insert(index_to_insert, merged_annotation)
        else:
            # Jeśli nie znaleziono odpowiedniego miejsca, dodaj na końcu
            self.sentence.tokens.append(merged_annotation)

        # Czyszczenie stanu
        self.labels_to_merge = deque()

        # Aktualizacja widżetu AnnotationContainer
        self.ids.annotation_container.sentence = self.sentence
        self.ids.annotation_container.update_tokens()

        print("Merged labels:", merged_annotation)
