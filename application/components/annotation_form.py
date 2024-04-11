from kivy.app import App
from kivy.core.window import Window
from kivy.lang import Builder
from kivy.properties import ObjectProperty, ListProperty
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
    labels = ListProperty([])

    def on_selected_label(self, instance, value):
        self.ids.current_annotation.clear_widgets()
        if value is not None:
            color_label = ColorLabel(label=value)
            color_label.on_touch_down = lambda touch: None
            color_label.border_color = color_label.border_color[:3] + [1]
            color_label.selected = 1
            color_label.size_hint = 1, 1
            self.ids.current_annotation.add_widget(color_label)
