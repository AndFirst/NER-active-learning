from kivy.app import App
from kivy.core.window import Window
from kivy.lang import Builder
from kivy.properties import ObjectProperty
from kivy.uix.boxlayout import BoxLayout

from application.components.annotation_box import AnnotationBox
from application.components.label import ColorLabel
from application.ui_colors import BACKGROUND_COLOR
from application.components.label_choose_container import LabelChooseContainer
from kivy.config import Config

Config.set("input", "mouse", "mouse,multitouch_on_demand")

kv_string = """
<AnnotationForm>:
    spacing: 10
    orientation: 'vertical'
    LabelChooseContainer:
        id: choose_container
    BoxLayout:
        id: current_annotation
        orientation: 'horizontal'
        size_hint: 1, 0.1
        canvas.before:
            Color:
                rgba: 0, 0, 0, 1 
            Line:
                rectangle: (self.x, self.y, self.width, self.height)
                width: 1 
    AnnotationBox:
        id: annotation_box
    BoxLayout:
        orientation: 'horizontal'
        size_hint: 1, 0.2   
        canvas.before:
            Color:
                rgba: 0, 0, 0, 1 
            Line:
                rectangle: (self.x, self.y, self.width, self.height)
                width: 1 
        Button:
            text: 'Akceptuj'
            font_size: '25sp'
            background_color: 0, 1, 0, 1  # Zielony kolor w formacie RGBA
        Button:
            text: 'Resetuj'
            font_size: '25sp'
            background_color: 1, 0, 0, 1  # Czerwony kolor w formacie RGBA
"""

Builder.load_string(kv_string)


# @TODO One label - multiple word - IREK
# @TODO Save on 'Akceptuj' click - IREK
# @TODO Reset on 'Reset' click - IREK
# @TODO Save on press exit button when there is unsaved progress. - IREK
class AnnotationForm(BoxLayout):
    selected_label = ObjectProperty(None, allownone=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ids.choose_container.annotation_form = self
        self.ids.annotation_box.annotation_form = self

    def on_selected_label(self, instance, value):
        self.ids.current_annotation.clear_widgets()
        if value is not None:
            color_label = ColorLabel(label=value)
            color_label.on_touch_down = lambda touch: None
            color_label.border_color = color_label.border_color[:3] + [1]
            color_label.selected = 1
            color_label.size_hint = 1, 1
            self.ids.current_annotation.add_widget(color_label)


class MyApp(App):
    def build(self):
        from kivy.modules import inspector

        Window.clearcolor = BACKGROUND_COLOR
        layout = AnnotationForm(padding=20)
        inspector.create_inspector(Window, layout)

        labels = [
            ("B-geo", 0, (0.8, 0.2, 0.2, 1)),  # Red for B-xxx
            ("B-per", 1, (0.2, 0.8, 0.2, 1)),  # Green for B-xxx
            ("B-org", 2, (0.2, 0.2, 0.8, 1)),  # Blue for B-xxx
            ("I-tim", 3, (0.8, 0.2, 0.2, 1)),  # Red for I-xxx
            ("I-org", 4, (0.2, 0.2, 0.8, 1)),  # Blue for I-xxx
            ("B-nat", 5, (0.8, 0.8, 0.2, 1)),  # Yellow for B-xxx
            ("I-eve", 6, (0.2, 0.8, 0.8, 1)),  # Cyan for I-xxx
            ("I-gpe", 7, (0.8, 0.2, 0.8, 1)),  # Purple for I-xxx
            ("B-art", 8, (0.8, 0.5, 0.2, 1)),  # Orange for B-xxx
            ("I-nat", 9, (0.8, 0.2, 0.5, 1)),  # Pink for I-xxx
            ("I-art", 10, (0.2, 0.8, 0.5, 1)),  # Greenish Blue for I-xxx
            ("B-gpe", 11, (0.5, 0.2, 0.8, 1)),  # Bluish Purple for B-xxx
            ("I-geo", 12, (0.2, 0.5, 0.8, 1)),  # Blueish Green for I-xxx
            ("B-tim", 13, (0.8, 0.5, 0.5, 1)),  # Light Red for B-xxx
            ("B-eve", 14, (0.5, 0.8, 0.5, 1)),  # Light Green for B-xxx
            ("I-per", 15, (0.5, 0.5, 0.8, 1)),  # Light Blue for I-xxx
            ("O", 16, (0, 0, 0, 0)),  # Black with 0 alpha for "O"
        ]

        data = [
            "Thousands",
            "of",
            "demonstrators",
            "have",
            "marched",
            "through",
            "London",
            "to",
            "protest",
            "the",
            "war",
            "in",
            "Iraq",
            "and",
            "demand",
            "the",
            "withdrawal",
            "of",
            "British",
            "troops",
            "from",
            "that",
            "country",
            ".",
        ]
        layout.ids.annotation_box.data = data  # Pass data to AnnotationBox
        layout.ids.choose_container.labels = labels
        return layout


if __name__ == "__main__":
    MyApp().run()
