from collections import deque

from kivy.app import App
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.properties import (
    ObjectProperty,
    ListProperty,
    BooleanProperty,
    StringProperty,
)
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.textinput import TextInput

from app.components.label import ColorLabel
from kivy.config import Config

from app.data_types import Annotation, Sentence, Word


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
        size_hint_y: 0.1
        orientation: 'horizontal'
        BoxLayout:
            id: current_annotation
            orientation: 'horizontal'
            size_hint_x: 0.5
            canvas.before:
                Color:
                    rgba: 0, 0, 0, 1 
                Line:
                    rectangle: (self.x, self.y, self.width, self.height)
                    width: 1 
        BoxLayout:
            size_hint_x: 0.25
            Button:
                id: ai_assistant_button
                text: "AI Assistant"
                font_size: '25sp'
                background_color: [0, 1, 0, 1] if root.ai_assistant_enabled else [1, 0, 0, 1]
                on_release: root.toggle_ai_assistant()
        BoxLayout:
            size_hint_x: 0.25
            Button:
                text: '?'
                font_size: '25sp'
                background_normal: ''
                background_color: 0, 0, 0, 0
                pos_hint: {'center_x': 0.5}
                canvas.before:
                    Color:
                        rgba: 0.5, 0.5, 0.5, 1
                    Ellipse:
                        pos: (self.pos[0] + self.width / 2 - self.height / 2, self.pos[1])
                        size: (self.height, self.height)
                on_release: root.show_instruction_popup()
    AnnotationContainer:
        id: annotation_container
        size_hint_y: 0.6
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
            on_release: root.accept()
        Button:
            text: 'Reset'
            font_size: '25sp'
            background_color: 1, 0, 0, 1
            on_release: root.reset()
        Button:
            id: multiword_mode_button
            
            text: 'Multiword Mode'
            font_size: '25sp'
            background_color: [0, 1, 0, 1] if root.multiword_mode else [1, 0, 0, 1]
            on_release: root.toggle_multiword_mode()
"""

Builder.load_string(kv_string)


# @TODO Save on press exit button when there is unsaved progress. - IREK
class AnnotationForm(BoxLayout):
    selected_label = ObjectProperty(None, allownone=True)
    labels = ListProperty([])
    sentence = ObjectProperty(None, allownone=True)
    labels_to_merge = ObjectProperty(deque(), allownone=True)
    multiword_mode = BooleanProperty(False)
    last_added_annotation = ObjectProperty(None, allownone=True)
    save_annotation_path = StringProperty("", allownone=False)

    ai_assistant_enabled = BooleanProperty(False)

    def toggle_ai_assistant(self):
        self.ai_assistant_enabled = not self.ai_assistant_enabled
        self.ids.ai_assistant_button.background_color = (
            [0, 1, 0, 1] if self.ai_assistant_enabled else [1, 0, 0, 1]
        )

    def accept(self):
        self.parent.parent.assistant.give_feedback(self.sentence)
        next_sentence = self.parent.parent.assistant.get_sentence(
            annotated=self.ai_assistant_enabled
        )
        if next_sentence:
            self.sentence = next_sentence
        else:

            self.sentence = None
            content = Label(
                text="All data has been annotated.\nYou're free now! Have a nice day!\nApplication will close now.",
            )
            popup = Popup(
                title="Annotation Complete",
                content=content,
                size_hint=(None, None),
                size=(400, 200),
            )
            popup.bind(on_dismiss=self.close_app)
            popup.open()

    def close_app(self, instance):
        def close(*args):
            App.get_running_app().stop()

        Clock.schedule_once(close, 0.5)

    def show_instruction_popup(self):
        instruction_text = (
            "BUTTONS\n"
            "Accept - accept current annotation and ask for next.\n"
            "Reset - reset all labels in text.\n"
            "Multiwords Mode"
            "   while green you can merge words into multiword annotation.\n"
            "       e.g. [United States] <- Geo\n"
            "       Left click add word to multiword annotation.\n"
            "       Right click ends current multiword annotation. \n"
            "   while red you can create only single word annotations.\n"
            "       e.g. [Poland] <- Geo\n"
            "       Left click create single word annotation from clicked word. \n"
            "       Right click removes label from clicked word. \n"
        )

        content_layout = BoxLayout(
            orientation="vertical", padding=20, spacing=10
        )

        instruction_input = TextInput(
            text=instruction_text, readonly=True, padding=(10, 10)
        )

        close_button = Button(
            text="Close", size_hint=(None, None), size=(100, 50)
        )
        close_button.bind(on_release=lambda btn: popup.dismiss())

        content_layout.add_widget(instruction_input)
        content_layout.add_widget(close_button)

        popup = Popup(
            title="Manual",
            content=content_layout,
            size_hint=(None, None),
            size=(600, 600),
        )
        popup.open()

    def toggle_multiword_mode(self):
        self.multiword_mode = not self.multiword_mode
        self.ids.multiword_mode_button.background_color = (
            [0, 1, 0, 1] if self.multiword_mode else [1, 0, 0, 1]
        )
        self.commit_multi_label()

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

    def update_selected_label(self, new_label: ColorLabel):
        for label_widget in self.ids.choose_container.children:
            if (
                isinstance(label_widget, ColorLabel)
                and label_widget != new_label
            ):
                label_widget.selected = 0

        new_label.selected = 1 - new_label.selected
        if new_label.selected:
            self.selected_label = new_label
        else:
            self.selected_label = None
        self.commit_multi_label()

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

    def remove_label(self, word: Word):
        token = self.sentence.get_word_parent(word)
        if len(token.words) == 1:
            token.label = None
        else:
            word_index = token.words.index(word)
            left_words = token.words[:word_index]
            right_words = token.words[word_index + 1 :]
            new_tokens = []
            if left_words:
                new_tokens.append(
                    Annotation(words=left_words, label=token.label)
                )
            new_tokens.append(Annotation(words=[word], label=None))
            if right_words:
                new_tokens.append(
                    Annotation(words=right_words, label=token.label)
                )
            index_to_replace = self.sentence.tokens.index(token)
            self.sentence.tokens.pop(index_to_replace)
            self.sentence.tokens[index_to_replace:index_to_replace] = (
                new_tokens
            )
        self.ids.annotation_container.sentence = self.sentence
        self.ids.annotation_container.update_tokens()

    def update_single_word_mode(self, word: Word):
        parent = self.sentence.get_word_parent(word)

        left_neighbor = self.sentence.get_left_neighbor(word)
        right_neighbor = self.sentence.get_right_neighbor(word)

        left_parent = self.sentence.get_word_parent(left_neighbor)
        right_parent = self.sentence.get_word_parent(right_neighbor)

        annotation = Annotation(
            words=[word], label=self.selected_label.label_data
        )

        left_index = (
            parent.words.index(left_neighbor)
            if left_parent == parent
            else None
        )
        right_index = (
            parent.words.index(right_neighbor)
            if right_parent == parent
            else None
        )
        new_tokens = []
        if left_index is not None:
            left_index += 1
            left_words = parent.words[:left_index]
            new_tokens.append(Annotation(words=left_words, label=parent.label))
        new_tokens.append(annotation)
        if right_index is not None:
            right_words = parent.words[right_index:]
            new_tokens.append(
                Annotation(words=right_words, label=parent.label)
            )
        index_to_replace = self.sentence.tokens.index(parent)
        self.sentence.tokens.pop(index_to_replace)
        self.sentence.tokens[index_to_replace:index_to_replace] = new_tokens

        self.ids.annotation_container.sentence = self.sentence
        self.ids.annotation_container.update_tokens()

    def update_multi_word_mode(self, word: Word):
        if not self.labels_to_merge:
            self.labels_to_merge.append(word)
        elif word == self.sentence.get_left_neighbor(self.labels_to_merge[0]):
            self.labels_to_merge.appendleft(word)
        elif word == self.sentence.get_right_neighbor(
            self.labels_to_merge[-1]
        ):
            self.labels_to_merge.append(word)
        else:
            return

        merged_annotation = Annotation(
            words=list(self.labels_to_merge),
            label=self.selected_label.label_data,
        )

        first_removed_word = self.labels_to_merge[0]
        last_removed_word = self.labels_to_merge[-1]

        parent = self.sentence.get_word_parent(word)
        left_neighbor = self.sentence.get_left_neighbor(first_removed_word)
        right_neighbor = self.sentence.get_right_neighbor(last_removed_word)

        for index, token in enumerate(self.sentence.tokens):
            if first_removed_word in token.words:
                index_to_insert = index
                if (
                    len(token.words) > 1
                    and right_neighbor not in token.words
                    or left_neighbor in token.words
                ):
                    index_to_insert += 1
                break

        new_tokens = []
        if self.sentence.get_word_parent(left_neighbor) == parent:
            left_index = parent.words.index(left_neighbor)
            left_words = parent.words[: left_index + 1]
            new_tokens.append(Annotation(words=left_words, label=parent.label))

        new_tokens.append(merged_annotation)

        if self.sentence.get_word_parent(right_neighbor) == parent:
            right_index = parent.words.index(right_neighbor)
            right_words = parent.words[right_index:]
            new_tokens.append(
                Annotation(words=right_words, label=parent.label)
            )

        index_to_replace = self.sentence.tokens.index(parent)
        self.sentence.tokens[index_to_replace:index_to_replace] = new_tokens
        self.sentence.tokens.remove(parent)
        if self.last_added_annotation:
            self.sentence.tokens.remove(self.last_added_annotation)
        self.last_added_annotation = merged_annotation

        self.ids.annotation_container.sentence = self.sentence
        self.ids.annotation_container.update_tokens()

    def commit_multi_label(self):
        self.last_added_annotation = None
        self.labels_to_merge = deque()
