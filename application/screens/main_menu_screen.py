from kivy.uix.screenmanager import Screen
from kivy.lang import Builder
from components.annotation_form import AnnotationForm

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
        shared_data = kwargs.pop('shared_data', None)
        super(MainMenuScreen, self).__init__(**kwargs)
        self.shared_data = shared_data

    def on_enter(self):
        super(MainMenuScreen, self).on_enter()
        if self.shared_data:
            self.ids.annotation_form.labels = self.shared_data.labels
