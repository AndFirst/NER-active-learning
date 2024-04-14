from kivy.app import App
from kivy.core.window import Window
from kivy.uix.screenmanager import Screen, ScreenManager
from kivy.lang import Builder
from application.components.annotation_form import AnnotationForm
from application.data_types import ProjectData, LabelData, Annotation, Sentence, Word
from application.ui_colors import BACKGROUND_COLOR

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
        shared_data = kwargs.pop('shared_data', None)
        super(MainMenuScreen, self).__init__(**kwargs)
        self.shared_data = shared_data
        self.ids.annotation_form.labels = shared_data.labels


class MyApp(App):

    def build(self):
        label1 = LabelData(label="Cat", color=(1, 0, 0, 1))
        label2 = LabelData(label="Dog", color=(0, 1, 0, 1))
        label3 = LabelData(label="Bird", color=(0, 0, 1, 1))

        shared_data = ProjectData(
            name="Animal Recognition Project",
            description="A project to recognize various animals",
            dataset_path="/path/to/dataset",
            labels=[label1, label2, label3]
        )
        screen_manager = ScreenManager()
        main_menu_screen = MainMenuScreen(name='main_menu', shared_data=shared_data)
        screen_manager.add_widget(main_menu_screen)
        data = [
            "A" * 10,
            "B" * 10,
            "C" * 10,
            "D" * 10,
            "E" * 10,
            "F" * 10,
            "G" * 10,
            "H" * 10,
            "I" * 10,
            "J" * 10,
        ]
        annotations = [Annotation(words=[Word(word)], label=None) for word in data]
        sentence = Sentence(tokens=annotations)

        main_menu_screen.ids.annotation_form.sentence = sentence
        return screen_manager


if __name__ == '__main__':
    MyApp().run()
