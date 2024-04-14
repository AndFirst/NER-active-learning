from kivy.lang import Builder
from kivy.properties import ObjectProperty, ListProperty
from kivy.uix.boxlayout import BoxLayout
from application.components.label import ColorLabel
from application.components.label_choose_container import LabelChooseContainer
from application.components.annotation_container import AnnotationContainer
from kivy.config import Config

from application.components.token import Token
from application.data_types import Annotation, Sentence

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
    labels_to_merge = ListProperty([], allownone=True)

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
        if self.selected_label:
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
                print(word)
                new_token = Annotation(words=[word], label=None)
                new_tokens.append(new_token)
        self.sentence = Sentence(tokens=new_tokens)
        self.selected_label = None
        self.ids.current_annotation.clear_widgets()
        for label in self.ids.choose_container.children:
            label.selected = 0
