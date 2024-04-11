from kivy.lang import Builder
from kivy.properties import ListProperty, ObjectProperty
from kivy.uix.label import Label
from kivy.uix.stacklayout import StackLayout

from application.components.label import ColorLabel
from application.data_types import LabelData

kv_string = """
<LabelChooseContainer>:
    size_hint: 1, 0.2
    padding: 10
    spacing: 10
    canvas.before:
        Color:
            rgba: 0, 0, 0, 1 
        Line:
            rectangle: (self.x, self.y, self.width, self.height)
            width: 1 
"""
Builder.load_string(kv_string)


class LabelChooseContainer(StackLayout):
    labels = ListProperty([])  # Initial data
    annotation_form = ObjectProperty(None)  # Added property to store annotation_form

    def __init__(self, **kwargs):
        super(LabelChooseContainer, self).__init__(**kwargs)
        self.bind(labels=self.on_labels_change)
        self.update_widgets()

    def update_widgets(self):
        self.clear_widgets()  # Clear existing widgets
        for label_data in self.labels:
            color_label = ColorLabel(label_data, annotation_form=self.annotation_form)
            self.add_widget(color_label)

    def on_labels_change(self, instance, value):
        self.update_widgets()
