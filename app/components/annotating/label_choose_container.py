from kivy.lang import Builder
from kivy.properties import ListProperty, ObjectProperty
from kivy.uix.stacklayout import StackLayout
from app.components.annotating.label import ColorLabel

kv_string = """
<LabelChooseContainer>:
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
    labels = ListProperty([])
    label_callback = ObjectProperty(None)

    # def __init__(self, callback**kwargs):
    #     super(LabelChooseContainer, self).__init__(**kwargs)

    def on_labels(self, *args):
        self.update_labels()

    def on_callback(self, *args):
        self.update_labels()

    def update_labels(self):
        self.clear_widgets()
        for label_data in self.labels:
            color_label = ColorLabel(
                label_data=label_data, update_form_state=self.label_callback
            )
            self.add_widget(color_label)
