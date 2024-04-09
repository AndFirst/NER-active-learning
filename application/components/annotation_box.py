from kivy.properties import ListProperty, ObjectProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.lang import Builder
from kivy.uix.label import Label
from kivy.uix.stacklayout import StackLayout
from application.components.annotation_word import AnnotationWord
from application.components.label import ColorLabel
from application.data_types import AnnotateLabel

kv_string = """
<AnnotationBox>:
    padding: 10
    spacing: 10
    size_hint: 1, 0.5
    canvas.before:
        Color:
            rgba: 0, 0, 0, 1 
        Line:
            rectangle: (self.x, self.y, self.width, self.height)
            width: 1
"""
Builder.load_string(kv_string)


class AnnotationBox(StackLayout):
    data = ListProperty([])  # Initial data
    annotation_form = ObjectProperty(None)

    def __init__(self, **kwargs):
        super(AnnotationBox, self).__init__(**kwargs)
        self.update_widgets()  # Update widgets based on initial data

    def update_widgets(self):
        self.clear_widgets()  # Clear existing widgets
        for annotation in self.data:
            annotate_label = AnnotateLabel(word=annotation, label=None)
            word = AnnotationWord(annotate_label, annotation_form=self.annotation_form)
            self.add_widget(word)  # Add word to current row

    def on_data(self, instance, value):
        self.update_widgets()  # Update widgets when data changes
