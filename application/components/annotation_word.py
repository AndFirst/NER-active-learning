from kivy.uix.boxlayout import BoxLayout
from kivy.lang import Builder
from kivy.properties import ListProperty, ObjectProperty

from application.data_types import AnnotateLabel

kv_string = """
<AnnotationWord>:
    size: self.minimum_size
    size_hint: None, None
    padding: 0
    canvas.before:
        Color:
            rgba: root.border_color
        Line:
            rectangle: (self.x, self.y, self.width, self.height)
            width: 1.5 
    orientation: 'vertical'
    Label:
        id: text_label
        font_size: '25sp'
        padding: 10, 0
        size_hint: None, None
        size: self.texture_size
        color: 0, 0, 0, 1
    BoxLayout:
        size_hint: 1, None    
        pos_hint: {'center_x': 0.5}
        size: self.minimum_size
        orientation: 'vertical'
        canvas.before:
            Color:
                rgba: root.border_color
            Rectangle:
                pos: self.pos
                size: self.size if root.show_border else (0, 0)
        Label:
            id: label_label
            font_size: '15sp'
            size_hint: None, None
            size: self.texture_size
            color: 0, 0, 0, 1
            pos_hint: {'center_x': 0.5}
            bold: True
"""

Builder.load_string(kv_string)


class AnnotationWord(BoxLayout):
    border_color = ListProperty([0, 0, 0, 0])
    show_border = True
    annotate_label = ObjectProperty(None)
    annotation_form = ObjectProperty(None)

    def __init__(self, annotate_label: AnnotateLabel, **kwargs):
        super(AnnotationWord, self).__init__(**kwargs)
        self.ids.text_label.text = annotate_label.word
        self.annotate_label = annotate_label
        self.ids.text_label.text = annotate_label.word
        self.update_label()

    def update_label(self):
        label = self.annotate_label.label
        if label and label.label != "O":
            self.ids.label_label.text = label.label
            self.show_border = True
            self.border_color = label.color

        else:
            self.ids.label_label.text = " "
            self.show_border = False
            self.border_color = [0, 0, 0, 0]

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            if self.annotation_form.selected_label:
                if touch.button == "left":
                    self.annotate_label.label = self.annotation_form.selected_label
                elif touch.button == "right":
                    self.annotate_label.label = None
                self.update_label()
                self.canvas.ask_update()
